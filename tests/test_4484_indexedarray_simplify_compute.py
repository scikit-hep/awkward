# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""CPU contract checks and host execution of the actual CCCL operator bodies.

Host substitutes do not test GPU compilation or execution; tests-cuda runs the
same case inventory through real CUDA dispatch.
"""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import awkward as ak
from awkward._backends.numpy import NumpyBackend

yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = {"awkward_IndexedArray_simplify": "awkward_IndexedArray_simplify"}

DTYPES = {"int32_t": np.int32, "uint32_t": np.uint32, "int64_t": np.int64}


def cases():
    specifications = yaml.safe_load((ROOT / "kernel-specification.yml").read_text())[
        "kernels"
    ]
    data = {
        x["name"]: x["tests"]
        for x in json.loads((ROOT / "kernel-test-data.json").read_text())["unit-tests"]
    }
    # Width-specific extremes are filtered per specialization below.
    data["awkward_IndexedArray_simplify"].extend(
        [
            {
                "inputs": {
                    "outerindex": [-1, -9, -9223372036854775808],
                    "innerindex": [],
                    "outerlength": 3,
                    "innerlength": 0,
                },
                "outputs": {"toindex": [-1, -1, -1, 123, 123, 123]},
                "error": False,
                "message": "",
            },
            {
                "inputs": {
                    "outerindex": [4294967295, 0],
                    "innerindex": [1],
                    "outerlength": 2,
                    "innerlength": 1,
                },
                "outputs": {"toindex": [123, 123, 123, 123, 123]},
                "error": True,
                "message": "index out of range",
            },
            {
                "inputs": {
                    "outerindex": [9223372036854775807, 0],
                    "innerindex": [1],
                    "outerlength": 2,
                    "innerlength": 1,
                },
                "outputs": {"toindex": [123, 123, 123, 123, 123]},
                "error": True,
                "message": "index out of range",
            },
            {
                "inputs": {
                    "outerindex": [0, 1, 99],
                    "innerindex": [4294967295, 2147483648],
                    "outerlength": 2,
                    "innerlength": 2,
                },
                "outputs": {"toindex": [4294967295, 2147483648, 123, 123, 123]},
                "error": False,
                "message": "",
            },
            {
                "inputs": {
                    "outerindex": [0, 1],
                    "innerindex": [-9223372036854775808, 9223372036854775807],
                    "outerlength": 2,
                    "innerlength": 2,
                },
                "outputs": {
                    "toindex": [
                        -9223372036854775808,
                        9223372036854775807,
                        123,
                        123,
                        123,
                    ]
                },
                "error": False,
                "message": "",
            },
        ]
    )
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
        **{name: array_factory(getattr(np, name)) for name in ("empty", "asarray")},
        int64=np.int64,
        cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(ptr=123)),
    )

    def transform(*, d_in, d_out, op, num_items, stream):
        assert stream.__cuda_stream__() == (0, 123)
        for i in range(num_items):
            value = op(np.int64(d_in.start + i))
            if not isinstance(d_out, _Discard):
                d_out[i] = value

    def reduce(*, d_in, d_out, num_items, h_init, **kwargs):
        assert isinstance(d_in, np.ndarray)
        assert d_in.dtype == np.dtype(np.int64)
        assert kwargs["stream"].__cuda_stream__() == (0, 123)
        d_out[0] = min([int(h_init), *(int(d_in[i]) for i in range(num_items))])

    tree = ast.parse((ROOT / "src/awkward/_connect/cuda/_compute.py").read_text())
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "awkward_IndexedArray_simplify"
    ]
    namespace = {
        "__name__": __name__,
        "np": np,
        "cp": cp,
        "SimpleNamespace": SimpleNamespace,
        "CountingIterator": _Counting,
        "DiscardIterator": _Discard,
        "OpKind": SimpleNamespace(MINIMUM=1),
        "unary_transform": transform,
        "reduce_into": reduce,
    }
    exec(
        compile(
            ast.Module(body=body, type_ignores=[]),
            str(ROOT / "src/awkward/_connect/cuda/_compute.py"),
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


@pytest.mark.parametrize("outer_dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("inner_dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("option", [False, True])
def test_public_simplification(compute_backend, outer_dtype, inner_dtype, option):
    inner_class = (
        ak.contents.IndexedOptionArray
        if option and inner_dtype != np.uint32
        else ak.contents.IndexedArray
    )
    inner_values = (
        [2, -1, 0] if inner_class is ak.contents.IndexedOptionArray else [2, 1, 0]
    )
    inner = inner_class(
        ak.index.Index(np.asarray(inner_values, dtype=inner_dtype)),
        ak.contents.NumpyArray(np.asarray([10, 20, 30])),
    )
    outer_values = [1, -1, 0] if option and outer_dtype != np.uint32 else [1, 0, 2]
    outer = ak.index.Index(np.asarray(outer_values, dtype=outer_dtype))
    outer_class = ak.contents.IndexedOptionArray if option else ak.contents.IndexedArray
    expected = outer_class.simplified(outer, inner).to_list()
    inner._backend = compute_backend
    assert outer_class.simplified(outer, inner).to_list() == expected
    with pytest.raises(ValueError, match="index out of range"):
        outer_class.simplified(
            ak.index.Index(np.asarray([1, 3], dtype=outer_dtype)), inner
        )


def large_case(length, error_at):
    rng = np.random.default_rng(2026)
    outer = rng.integers(-1, 64, size=length, dtype=np.int64)
    if error_at is not None:
        outer[error_at] = 64
        if error_at + 1 < length:
            outer[error_at + 1] = 99
    inner = rng.integers(-7, 100, size=64, dtype=np.int64)
    return [np.full(length + 7, 123, dtype=np.int64), outer, length, inner, inner.size]


@pytest.mark.parametrize(
    "length,error_at", [(33, None), (257, 0), (4097, 256), (4097, 4096)]
)
def test_first_error_across_blocks(implementation, length, error_at):
    args = large_case(length, error_at)
    expected_args = copy.deepcopy(args)
    expected = NumpyBackend.instance()[
        ("awkward_IndexedArray_simplify", np.int64, np.int64, np.int64)
    ](*expected_args)
    actual = implementation["awkward_IndexedArray_simplify"](*args)
    assert_error(actual, expected)
    np.testing.assert_array_equal(args[0], expected_args[0])


@pytest.fixture(scope="module")
def prepared_implementation():
    import weakref
    from functools import lru_cache
    from threading import RLock

    registry = weakref.WeakValueDictionary()
    events = []

    class Buffer(_DeviceArray):
        @property
        def __cuda_array_interface__(self):
            registry[self.ctypes.data] = self
            return {
                "shape": self.shape,
                "strides": None,
                "typestr": self.dtype.str,
                "data": (self.ctypes.data, False),
                "version": 3,
            }

    def prepare(**kwargs):
        events.append(("prepare",))
        operation = kwargs["op"]

        def launch(**kwargs):
            import time

            time.sleep(0.0001)
            events.append(("transform", kwargs["num_items"]))
            for i in range(kwargs["num_items"]):
                value = operation(i)
                if not isinstance(kwargs["d_out"], _Discard):
                    kwargs["d_out"][i] = value

        return launch

    def prepare_reduce(**kwargs):
        events.append(("prepare_reduce",))

        def launch(*, temp_storage, **kwargs):
            if temp_storage is None:
                return 64
            events.append(("reduce", kwargs["num_items"]))
            kwargs["d_out"][0] = min(
                [int(kwargs["h_init"]), *kwargs["d_in"][: kwargs["num_items"]]]
            )
            return 64

        return launch

    def empty(length, dtype):
        events.append(("allocate", length, np.dtype(dtype).str))
        return np.empty(length, dtype=dtype).view(Buffer)

    namespace = {
        "np": np,
        "cp": SimpleNamespace(
            empty=empty,
            int64=np.int64,
            uint8=np.uint8,
            cuda=SimpleNamespace(
                get_current_stream=lambda: SimpleNamespace(ptr=123),
                runtime=SimpleNamespace(getDevice=lambda: 0),
                driver=SimpleNamespace(ctxGetCurrent=lambda: 1),
            ),
        ),
        "lru_cache": lru_cache,
        "_lock": RLock(),
        "CountingIterator": _Counting,
        "DiscardIterator": _Discard,
        "OpKind": SimpleNamespace(MINIMUM=1),
        "make_op_adapter": lambda op: op,
        "make_unary_transform": prepare,
        "make_reduce_into": prepare_reduce,
    }
    path = ROOT / "src/awkward/_connect/cuda/_indexedarray.py"
    tree = ast.parse(path.read_text())
    body = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    state = namespace["_ArrayState"]
    state.__getitem__ = lambda self, i: registry[self.data_ptr()][i]
    state.__setitem__ = lambda self, i, value: registry[self.data_ptr()].__setitem__(
        i, value
    )
    return namespace, Buffer, events


@pytest.mark.parametrize("name,specialization,case", CASES)
def test_prepared_cpu_parity(name, specialization, case, prepared_implementation):
    namespace, buffer, _ = prepared_implementation
    args, outputs, key = arguments(specialization, case)
    expected_args = copy.deepcopy(args)
    expected = NumpyBackend.instance()[(name, *key)](*expected_args)
    device = [x.copy().view(buffer) if isinstance(x, np.ndarray) else x for x in args]
    if args[2] == 0:
        return
    row, attempt = namespace["simplify_status"](*device).get().tolist()
    assert (row < args[2]) == bool(expected.str)
    if expected.str:
        assert (row, attempt) == (expected.id, expected.attempt)
    for i in outputs:
        np.testing.assert_array_equal(device[i], expected_args[i])


@pytest.mark.parametrize("length", [1, 32, 64, 65, 129, 4097])
@pytest.mark.parametrize("error_row", [None, 0, 63, 64, 128])
def test_prepared_partial_output_and_scratch(
    length, error_row, prepared_implementation
):
    namespace, buffer, events = prepared_implementation
    outer = (np.arange(length, dtype=np.int64) % 3).view(buffer)
    inner = np.asarray([-7, 10, 20], dtype=np.int64).view(buffer)
    if error_row is not None and error_row < length:
        outer[error_row] = 3
        outer[-1] = 99
    output = np.full(length + 3, 123, dtype=np.int64).view(buffer)
    expected = output.copy()
    error = NumpyBackend.instance()[
        "awkward_IndexedArray_simplify", np.int64, np.int64, np.int64
    ](expected, outer, length, inner, len(inner))
    events.clear()
    row, attempt = namespace["simplify_status"](
        output, outer, length, inner, len(inner)
    ).get()
    np.testing.assert_array_equal(output, expected)
    if error.str:
        assert (row, attempt) == (error.id, error.attempt)
    else:
        assert row == length
    transforms = [event for event in events if event[0] == "transform"]
    if length <= 64:
        assert transforms == [("transform", 1)]
        assert not any(event[0] == "reduce" for event in events)
    else:
        chunks = (length + 63) // 64
        assert transforms == [("transform", chunks), ("transform", length)]
        assert ("allocate", chunks, np.dtype(np.int64).str) in events


def test_prepared_reuse_lifetimes_and_concurrent_calls(prepared_implementation):
    import gc
    import weakref
    from concurrent.futures import ThreadPoolExecutor

    namespace, buffer, events = prepared_implementation

    def run(value):
        outer = (np.arange(129, dtype=np.int64) % 3).view(buffer)
        inner = np.asarray([value, -7, 20], dtype=np.int64).view(buffer)
        output = np.full(132, 123, dtype=np.int64).view(buffer)
        if value % 2:
            outer[64] = 3
            outer[100] = 99
        expected = output.copy()
        error = NumpyBackend.instance()[
            "awkward_IndexedArray_simplify", np.int64, np.int64, np.int64
        ](expected, outer, 129, inner, 3)
        references = [weakref.ref(x) for x in (outer, inner, output)]
        row, attempt = namespace["simplify_status"](output, outer, 129, inner, 3).get()
        np.testing.assert_array_equal(output, expected)
        if error.str:
            assert (row, attempt) == (error.id, error.attempt)
        else:
            assert row == 129
        return references

    run(0)
    events.clear()
    with ThreadPoolExecutor(max_workers=4) as executor:
        references = list(executor.map(run, range(12)))
    gc.collect()
    assert all(reference() is None for group in references for reference in group)
    assert not any(event[0].startswith("prepare") for event in events)
