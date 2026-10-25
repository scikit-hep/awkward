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
    "awkward_ListOffsetArray_flatten_offsets": "flatten_offsets",
    "awkward_ListOffsetArray_toRegularArray": "to_regular",
    "awkward_ListOffsetArray_rpad_axis1": "rpad",
    "awkward_ListOffsetArray_drop_none_indexes": "drop_none",
    "awkward_ListOffsetArray_reduce_local_nextparents_64": "local_parents",
    "awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64": "nonlocal_ranges",
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
            for name in ("empty", "zeros", "full", "asarray", "arange")
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

    def segmented(
        *, d_in, d_out, num_segments, start_offsets_in, end_offsets_in, h_init, **kwargs
    ):
        assert kwargs["stream"].__cuda_stream__() == (0, 123)
        # Exercise the supported array-input path, not a stateful iterator.
        assert isinstance(d_in, np.ndarray)
        assert d_in.dtype == np.dtype(np.int64)
        assert kwargs["max_segment_size"] is not None
        for row in range(num_segments):
            values = [
                d_in[i] for i in range(start_offsets_in[row], end_offsets_in[row])
            ]
            d_out[row] = max([int(h_init), *values])

    tree = ast.parse(
        (ROOT / "src/awkward/_connect/cuda/_listoffsetarray.py").read_text()
    )
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
        "segmented_reduce": segmented,
        "DiscardIterator": _Discard,
        "OpKind": SimpleNamespace(PLUS=0, MINIMUM=1, MAXIMUM=2),
        "unary_transform": transform,
        "inclusive_scan": scan,
        "reduce_into": reduce,
    }
    exec(
        compile(
            ast.Module(body=body, type_ignores=[]),
            str(ROOT / "src/awkward/_connect/cuda/_listoffsetarray.py"),
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
    if name not in {
        "awkward_ListOffsetArray_toRegularArray",
        "awkward_ListOffsetArray_drop_none_indexes",
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


@pytest.mark.parametrize("offsets", [[2, 4, 6], [0, 0, 0], [0], [0, 2, 5]])
def test_public_regular_conversion(compute_backend, offsets):
    layout = ak.contents.ListOffsetArray(
        ak.index.Index64(np.asarray(offsets)),
        ak.contents.NumpyArray(np.arange(6)),
    )
    expected = ak.Array(layout).to_list()
    layout._backend = compute_backend
    if offsets == [0, 2, 5]:
        with pytest.raises(ValueError, match="not regular"):
            ak.to_regular(layout)
    else:
        assert ak.to_regular(layout).to_list() == expected


@pytest.mark.parametrize("groups,width", [(1, 1), (17, 33), (257, 513)])
def test_nonlocal_ranges_across_block_boundaries(implementation, groups, width):
    rng = np.random.default_rng(2026)
    distincts = rng.choice([-1, -2, 0, 7], groups * width).astype(np.int64)
    starts = np.full(groups + 3, 123, dtype=np.int64)
    stops = starts.copy()
    expected_starts, expected_stops = starts.copy(), stops.copy()
    kernel = NumpyBackend.instance()[
        "awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64",
        np.int64,
        np.int64,
        np.int64,
    ]
    kernel(expected_starts, expected_stops, distincts, distincts.size, groups)
    implementation["nonlocal_ranges"](starts, stops, distincts, distincts.size, groups)
    np.testing.assert_array_equal(starts, expected_starts)
    np.testing.assert_array_equal(stops, expected_stops)
