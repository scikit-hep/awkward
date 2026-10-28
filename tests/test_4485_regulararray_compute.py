# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""CPU contract checks and host execution of the actual CCCL operator bodies.

Host substitutes do not test GPU compilation or execution; tests-cuda runs the
same case inventory through real CUDA dispatch.
"""

import ast
import copy
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest

import awkward as ak
from awkward._backends.numpy import NumpyBackend

yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = {
    "awkward_RegularArray_getitem_next_array_regularize": "regularize",
    "awkward_RegularArray_reduce_local_nextparents_64": "local_parents",
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
        uint8=np.uint8,
        maximum=np.maximum,
        cuda=SimpleNamespace(
            get_current_stream=lambda: SimpleNamespace(ptr=123),
            runtime=SimpleNamespace(getDevice=lambda: 0),
            driver=SimpleNamespace(ctxGetCurrent=lambda: 1),
        ),
    )

    def transform(*, d_in, d_out, op, num_items, stream):
        assert stream.__cuda_stream__() == (0, 123)
        for i in range(num_items):
            value = op(np.int64(d_in.start + i))
            if not isinstance(d_out, _Discard):
                d_out[i] = value

    def reduce(*, d_in, d_out, num_items, h_init, **kwargs):
        assert kwargs["stream"].__cuda_stream__() == (0, 123)
        d_out[0] = min(int(h_init), *(int(x) for x in d_in[:num_items]))

    def make_transform(**kwargs):
        return transform

    def make_reduce(**kwargs):
        def prepared(*, temp_storage, d_in, d_out, op, num_items, h_init, stream):
            if temp_storage is None:
                return 16
            value = np.int64(h_init)
            for row in range(num_items):
                value = op(value, np.int64(row))
            d_out[0] = value

        return prepared

    buffers = {}

    def store(address, index, value):
        output = buffers[int(address)]
        assert 0 <= index < output.size
        output[index] = value

    tree = ast.parse((ROOT / "src/awkward/_connect/cuda/_regulararray.py").read_text())
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name != "_store_i64"
    ]
    namespace = {
        "__name__": __name__,
        "np": np,
        "math": math,
        "numba_cuda": SimpleNamespace(jit=lambda **kwargs: lambda function: function),
        "_store_i64": store,
        "buffers": buffers,
        "cp": cp,
        "dataclass": dataclass,
        "lru_cache": lru_cache,
        "RLock": RLock,
        "_regularize_lock": RLock(),
        "make_op_adapter": lambda op: op,
        "make_unary_transform": make_transform,
        "make_reduce_into": make_reduce,
        "_SLICE_NONE": ak._util.kSliceNone,
        "CountingIterator": _Counting,
        "DiscardIterator": _Discard,
        "OpKind": SimpleNamespace(PLUS=0, MINIMUM=1, MAXIMUM=2),
        "unary_transform": transform,
        "reduce_into": reduce,
    }
    exec(
        compile(
            ast.Module(body=body, type_ignores=[]),
            str(ROOT / "src/awkward/_connect/cuda/_regulararray.py"),
            "exec",
        ),
        namespace,
    )
    state = namespace["_ArrayState"]
    state.bind = lambda self, array: setattr(self, "array", array)
    state.clear = lambda self: setattr(self, "array", None)
    state.__getitem__ = lambda self, index: self.array[index]
    state.__setitem__ = lambda self, index, value: self.array.__setitem__(index, value)
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
    if name == "awkward_RegularArray_reduce_local_nextparents_64":
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
    assert not any(
        name.startswith("awkward_RegularArray_")
        for name in ast.literal_eval(node.value)
    )


def combination_arguments(n, replacement, size, length):
    width = size + n - 1 if replacement else size
    count = math.comb(width, n) if width >= n and n >= 0 else 0
    total = length * count
    # Pointer targets are views with prefix and tail sentinels.
    backing = [np.full(total + 7, -99, dtype=np.int64) for _ in range(n)]
    carries = [x[2:] for x in backing]
    pointers = np.asarray([x.ctypes.data for x in carries], dtype=np.intp)
    args = [
        pointers,
        np.full(n + 3, -99, dtype=np.int64),
        np.arange(n + 3, dtype=np.int64) + 17,
        n,
        replacement,
        size,
        length,
    ]
    return args, backing, carries


@pytest.mark.parametrize("n", range(1, 6))
@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("size", range(7))
@pytest.mark.parametrize("length", [0, 1, 3])
def test_combinations_cpu_parity(implementation, n, replacement, size, length):
    args, backing, carries = combination_arguments(n, replacement, size, length)
    expected, expected_backing, _ = combination_arguments(n, replacement, size, length)
    key = ("awkward_RegularArray_combinations_64", np.int64, np.int64, np.int64)
    assert NumpyBackend.instance()[key](*expected).str is None
    implementation["buffers"].clear()
    implementation["buffers"].update((x.ctypes.data, x) for x in carries)
    _DeviceArray.reads = 0
    implementation["combinations"](*args)
    assert _DeviceArray.reads == 0
    for actual, wanted in zip(backing, expected_backing, strict=True):
        np.testing.assert_array_equal(actual, wanted)
    np.testing.assert_array_equal(args[1], expected[1])
    np.testing.assert_array_equal(args[2], expected[2])


def test_empty_combinations(implementation):
    args, _, _ = combination_arguments(0, False, 0, 0)
    before = copy.deepcopy(args)
    implementation["combinations"](*args)
    for i in range(3):
        np.testing.assert_array_equal(args[i], before[i])


def test_pointer_store_uses_mlir_device_registration():
    tree = ast.parse((ROOT / "src/awkward/_connect/cuda/_regulararray.py").read_text())
    helpers = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    for name in ("_choose", "_store_i64"):
        decorator = helpers[name].decorator_list[0]
        assert ast.unparse(decorator) == "numba_cuda.jit(device=True)"
    assert "ctypes.cast" in ast.unparse(helpers["_store_i64"])


@pytest.mark.parametrize(
    "top,choose", [(0, 0), (10, 0), (10, 10), (1000, 3), (1000, 999), (66, 33)]
)
def test_binomial_coefficients(implementation, top, choose):
    assert implementation["_choose"](top, choose) == math.comb(top, choose)


def test_binomial_native_lowering():
    numba = pytest.importorskip("numba")
    from numba.extending import register_jitable

    source = ROOT / "src/awkward/_connect/cuda/_regulararray.py"
    node = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_choose"
    )
    namespace = {
        "numba_cuda": SimpleNamespace(jit=lambda **kwargs: register_jitable),
        "np": np,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    compiled = numba.njit(namespace["_choose"])
    for top, choose in [(0, 0), (1000, 3), (66, 33)]:
        assert compiled(np.int64(top), np.int64(choose)) == math.comb(top, choose)


@pytest.mark.parametrize("top,choose", [(2, 3), (3, -1)])
def test_empty_binomial_coefficients(implementation, top, choose):
    assert implementation["_choose"](top, choose) == 0


@pytest.mark.parametrize("length", [1, 64, 65, 129, 4096])
@pytest.mark.parametrize("bad", [None, 0, -1])
def test_prepared_regularize_prefix_and_rebinding(implementation, length, bad):
    regularize = implementation["regularize"]
    for shift in (0, 1):
        source = (np.arange(length, dtype=np.int64) + shift) % 8 - 4
        if bad is not None:
            source[bad] = -9
            if bad == 0 and length > 1:
                source[-1] = 10
        expected = np.full(length + 3, -123, dtype=np.int64)
        actual = expected.copy()
        cpu = NumpyBackend.instance()[
            "awkward_RegularArray_getitem_next_array_regularize", np.int64, np.int64
        ]
        error = cpu(expected, source, length, 8)
        result = regularize(actual, source, length, 8)
        np.testing.assert_array_equal(actual, expected)
        assert_error(result, error)
    setup = implementation["_regularize_setup"](0, 1, actual.dtype.str, length, 8)
    assert setup.source.array is None and setup.output.array is None
    assert setup.status.array is None
    if length > 64:
        assert setup.storage_bytes == 16
