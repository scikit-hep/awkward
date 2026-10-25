# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""CPU contract checks and host execution of the actual CCCL operator bodies.

Host substitutes do not test GPU compilation or execution; tests-cuda runs the
same case inventory through real CUDA dispatch.
"""

import ast
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import awkward as ak
from awkward._backends.numpy import NumpyBackend

yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = {
    "awkward_ListArray_validity": "validity",
    "awkward_ListArray_getitem_next_at": "next_at",
    "awkward_ListArray_getitem_next_array": "next_array",
    "awkward_ListArray_getitem_next_array_advanced": "next_array_advanced",
    "awkward_ListArray_getitem_jagged_expand": "jagged_expand",
    "awkward_ListArray_getitem_jagged_apply": "jagged_apply",
    "awkward_ListArray_getitem_jagged_shrink": "jagged_shrink",
    "awkward_ListArray_rpad_axis1": "rpad",
    "awkward_ListArray_getitem_next_range_spreadadvanced": "spreadadvanced",
    "awkward_ListArray_getitem_next_range": "next_range",
}
DTYPES = {"int32_t": np.int32, "uint32_t": np.uint32, "int64_t": np.int64}


def cases():
    specifications = yaml.safe_load((ROOT / "kernel-specification.yml").read_text())[
        "kernels"
    ]
    data = {
        x["name"]: x["tests"]
        for x in json.loads((ROOT / "kernel-test-data.json").read_text())["unit-tests"]
    }
    output = []
    for spec in specifications:
        if spec["name"] not in FUNCTIONS:
            continue
        for specialization in spec["specializations"]:
            for i, case in enumerate(data[spec["name"]]):
                valid = True
                for argument in specialization["args"]:
                    if "List[" in argument["type"]:
                        dtype = next(
                            v for k, v in DTYPES.items() if k in argument["type"]
                        )
                        values = case["inputs"].get(
                            argument["name"], case["outputs"].get(argument["name"], [])
                        )
                        limits = np.iinfo(dtype)
                        if any(x < limits.min or x > limits.max for x in values):
                            valid = False
                if valid:
                    output.append(
                        pytest.param(
                            spec["name"],
                            specialization,
                            case,
                            id=f"{specialization['name']}-{i}",
                        )
                    )
    return output


CASES = cases()


def arguments(specialization, case):
    values = []
    outputs = []
    key = []
    for argument in specialization["args"]:
        name = argument["name"]
        if "List[" in argument["type"]:
            dtype = next(v for k, v in DTYPES.items() if k in argument["type"])
            key.append(dtype)
            if argument["dir"] == "out":
                sentinel = np.iinfo(dtype).max - 2
                value = np.full(len(case["outputs"][name]) + 8, sentinel, dtype=dtype)
                outputs.append(len(values))
            else:
                value = np.array(case["inputs"][name], dtype=dtype)
            values.append(value)
        else:
            values.append(case["inputs"][name])
    return values, outputs, key


class _DeviceArray(np.ndarray):
    reads = 0

    def get(self):
        type(self).reads += 1
        return np.asarray(self).copy()


class _Counting:
    def __init__(self, start):
        self.start = start


class _Discard:
    pass


def host_implementation():
    def array_factory(function):
        def allocate(*args, **kwargs):
            return function(*args, **kwargs).view(_DeviceArray)

        return allocate

    cp = SimpleNamespace(
        **{
            name: array_factory(getattr(np, name))
            for name in ("empty", "zeros", "full", "asarray")
        },
        int64=np.int64,
        maximum=np.maximum,
        cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(ptr=123)),
    )

    def transform(*, d_in, d_out, op, num_items, stream):
        assert stream.__cuda_stream__() == (0, 123)
        for i in range(num_items):
            value = op(np.int64(d_in.start + i))
            if not isinstance(d_out, _Discard):
                d_out[i] = value

    def scan(*, d_in, d_out, num_items, **kwargs):
        assert kwargs["stream"].__cuda_stream__() == (0, 123)
        d_out[:num_items] = np.cumsum(d_in[:num_items])

    def reduce(*, d_in, d_out, num_items, h_init, **kwargs):
        assert kwargs["stream"].__cuda_stream__() == (0, 123)
        d_out[0] = min(int(h_init), *(int(x) for x in d_in[:num_items]))

    tree = ast.parse((ROOT / "src/awkward/_connect/cuda/_listarray.py").read_text())
    body = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    namespace = {
        "__name__": __name__,
        "np": np,
        "cp": cp,
        "dataclass": dataclass,
        "_SLICE_NONE": ak._util.kSliceNone,
        "CountingIterator": _Counting,
        "DiscardIterator": _Discard,
        "OpKind": SimpleNamespace(PLUS=0, MINIMUM=1),
        "unary_transform": transform,
        "inclusive_scan": scan,
        "reduce_into": reduce,
    }
    exec(
        compile(
            ast.Module(body=body, type_ignores=[]),
            str(ROOT / "src/awkward/_connect/cuda/_listarray.py"),
            "exec",
        ),
        namespace,
    )
    return namespace


@pytest.fixture(scope="module")
def implementation():
    return host_implementation()


def assert_error(actual, expected):
    assert (actual is None or actual.str is None) == (
        expected is None or expected.str is None
    )
    if expected is not None and expected.str is not None:
        assert actual.str == expected.str
        assert actual.id == expected.id
        assert actual.attempt == expected.attempt


@pytest.mark.parametrize("name,specialization,case", CASES)
def test_cpu_parity(name, specialization, case, implementation):
    args, outputs, key = arguments(specialization, case)
    expected_args = copy.deepcopy(args)
    expected = NumpyBackend.instance()[(name, *key)](*expected_args)
    assert bool(expected.str) == bool(case["error"])
    _DeviceArray.reads = 0
    actual = implementation[FUNCTIONS[name]](*args)
    assert_error(actual, expected)
    for i in outputs:
        np.testing.assert_array_equal(args[i], expected_args[i])
    assert _DeviceArray.reads <= 1
    if name in {
        "awkward_ListArray_getitem_next_range",
        "awkward_ListArray_getitem_next_range_spreadadvanced",
        "awkward_ListArray_getitem_jagged_shrink",
        "awkward_ListArray_rpad_axis1",
    }:
        assert _DeviceArray.reads == 0


def test_raw_dispatch_removed():
    tree = ast.parse((ROOT / "dev/generate-kernel-signatures.py").read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "cuda_kernels_impl" for t in n.targets
        )
    )
    assert FUNCTIONS.keys().isdisjoint(ast.literal_eval(node.value))


@pytest.fixture(scope="module")
def compute_backend(implementation):
    class ComputeOnHost(NumpyBackend):
        def _new_kernel(self, key):
            if key[0] in FUNCTIONS:
                return implementation[FUNCTIONS[key[0]]]
            return super()._new_kernel(key)

    backend = object.__new__(ComputeOnHost)
    backend.__init__()
    return backend


@pytest.mark.parametrize("offsets", [False, True])
@pytest.mark.parametrize("exception", [False, True])
def test_public_validity_contract(compute_backend, offsets, exception):
    content = ak.contents.NumpyArray(np.array([1]))
    if offsets:
        layout = ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 2])), content
        )
    else:
        layout = ak.contents.ListArray(
            ak.index.Index64(np.array([0])), ak.index.Index64(np.array([2])), content
        )
    layout._backend = compute_backend
    if exception:
        with pytest.raises(ValueError, match="stop"):
            ak.validity_error(layout, exception=True)
        with pytest.raises(ValueError, match="stop"):
            ak.is_valid(layout, exception=True)
    else:
        assert "stop" in ak.validity_error(layout)
        assert ak.is_valid(layout) is False


@pytest.mark.parametrize("selector", [0, [0, 99], [[0], [0], [0]]])
def test_public_index_error_contract(compute_backend, selector):
    layout = ak.contents.ListArray(
        ak.index.Index64(np.array([0, 2, 2])),
        ak.index.Index64(np.array([2, 2, 4])),
        ak.contents.NumpyArray(np.arange(4)),
    )
    layout._backend = compute_backend
    with pytest.raises(IndexError):
        ak.Array(layout)[:, selector]
