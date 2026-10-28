# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

import awkward as ak
from awkward._backends.numpy import NumpyBackend

PAIRS = [
    (out, inp)
    for out in (np.int32, np.uint32, np.int64, np.uint64)
    for inp in (
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    )
    if np.dtype(out).kind == np.dtype(inp).kind
    and np.dtype(inp).itemsize <= np.dtype(out).itemsize
]


def inputs(dtype):
    info = np.iinfo(dtype)
    rows = [
        [info.max, 1],
        [info.min, -1 if info.min < 0 else 0],
        [info.max, 2],
        [info.min, 2],
        [info.max] * 10,
        [],
    ]
    values = np.asarray([99] + [x for row in rows for x in row] + [88], dtype=dtype)
    offsets = np.asarray(
        [1] + [1 + sum(map(len, rows[: i + 1])) for i in range(len(rows))],
        dtype=np.int64,
    )
    return rows, values, offsets


def expected(rows, out, op):
    bits = np.dtype(out).itemsize * 8
    modulus = 1 << bits
    result = []
    for row in rows:
        value = (sum(row) if op == "sum" else math.prod(row)) % modulus
        if np.dtype(out).kind == "i" and value >= modulus // 2:
            value -= modulus
        result.append(value)
    return result


@pytest.mark.parametrize("out,inp", PAIRS)
@pytest.mark.parametrize("op", ["sum", "prod"])
def test_kernel_wrapping(out, inp, op):
    rows, values, offsets = inputs(inp)
    result = np.empty(len(rows), dtype=out)
    kernel = NumpyBackend.instance()[f"awkward_reduce_{op}", out, inp, np.int64]
    error = kernel(result, values, offsets, len(rows))
    assert error.str is None
    np.testing.assert_array_equal(result, expected(rows, out, op))


@pytest.fixture(scope="module")
def reference():
    path = Path(__file__).resolve().parents[1] / "awkward-cpp/tests-spec/kernels.py"
    if not path.exists():
        pytest.skip("requires generated kernel references")
    spec = importlib.util.spec_from_file_location("overflow_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("out,inp", PAIRS)
@pytest.mark.parametrize("op", ["sum", "prod"])
def test_typed_reference_wrapping(reference, out, inp, op):
    rows, values, offsets = inputs(inp)
    result = {}
    func = getattr(
        reference, f"awkward_reduce_{op}_{np.dtype(out).name}_{np.dtype(inp).name}_64"
    )
    func(result, values.tolist(), offsets.tolist(), len(rows))
    assert list(result.values()) == expected(rows, out, op)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("op", ["sum", "prod"])
def test_public_wrapping(dtype, op):
    info = np.iinfo(dtype)
    values = np.asarray([[info.max, 1], [info.min, -1], [info.max, 2]], dtype=dtype)
    actual = ak.to_numpy(getattr(ak, op)(ak.Array(values), axis=-1))
    target = getattr(np, op)(values, axis=-1, dtype=actual.dtype)
    np.testing.assert_array_equal(actual, target)
    assert (
        getattr(ak, op)(ak.Array(values).layout.to_typetracer(), axis=-1).layout.form
        == ak.Array(actual).layout.form
    )


@pytest.fixture(scope="module")
def cuda_host_callbacks():
    """Execute real CUDA orchestration/cast bodies; this does not test GPU JIT."""
    import ast
    import functools
    from types import SimpleNamespace

    path = Path(ak.__file__).parent / "_connect/cuda/_compute.py"
    names = {
        "_make_widening_cast",
        "_widen_for_reduce",
        "_integer_reduction_buffers",
        "make_segment_views",
        "awkward_reduce_sum",
        "awkward_reduce_prod",
    }
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    calls = []

    class Transform:
        def __init__(self, values, op):
            self.values, self.op = values, op

        def __getitem__(self, index):
            return self.op(self.values[index])

    def reduce(**kwargs):
        calls.append(kwargs)
        output = kwargs["d_out"]
        operation = kwargs["op"]
        for row, (start, stop) in enumerate(
            zip(kwargs["start_offsets_in"], kwargs["end_offsets_in"], strict=True)
        ):
            value = kwargs["h_init"].item()
            for i in range(start, stop):
                item = np.asarray(kwargs["d_in"][i]).item()
                value = value + item if operation == "sum" else operation(value, item)
            if output.dtype.kind in "iu":
                value %= 1 << (8 * output.dtype.itemsize)
            output[row] = value

    namespace = {
        "__name__": "awkward._connect.cuda._compute",
        "np": np,
        "functools": functools,
        "TransformIterator": Transform,
        "OpKind": SimpleNamespace(PLUS="sum"),
        "segmented_reduce": reduce,
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace, calls=calls)


@pytest.mark.parametrize("out,inp", PAIRS)
@pytest.mark.parametrize("op", ["sum", "prod"])
def test_cuda_host_unsigned_accumulation(cuda_host_callbacks, out, inp, op):
    rows, values, offsets = inputs(inp)
    output = np.full(len(rows) + 2, 77, dtype=out)
    target = output[: len(rows)]
    function = getattr(cuda_host_callbacks, f"awkward_reduce_{op}")
    function(target, values, offsets, len(rows))
    np.testing.assert_array_equal(target, expected(rows, out, op))
    np.testing.assert_array_equal(output[-2:], [77, 77])
    submitted = cuda_host_callbacks.calls[-1]
    assert submitted["d_out"].dtype.kind == "u"
    assert np.shares_memory(submitted["d_out"], output)
    assert submitted["num_segments"] == len(rows)
    np.testing.assert_array_equal(submitted["start_offsets_in"], offsets[:-1])


@pytest.mark.parametrize("op", ["sum", "prod"])
def test_cuda_host_float_accumulation_unchanged(cuda_host_callbacks, op):
    values = np.array([1.5, 2.0, -3.0], dtype=np.float32)
    offsets = np.array([0, 2, 2, 3], dtype=np.int64)
    result = np.empty(3, dtype=np.float64)
    getattr(cuda_host_callbacks, f"awkward_reduce_{op}")(result, values, offsets, 3)
    np.testing.assert_array_equal(result, [3.5, 0, -3] if op == "sum" else [3, 1, -3])
    assert cuda_host_callbacks.calls[-1]["d_out"] is result
    cast = cuda_host_callbacks._make_widening_cast(np.float32, np.float64)
    assert cast is cuda_host_callbacks._make_widening_cast(np.float32, np.float64)
    assert cast.__annotations__ == {"x": np.float32, "return": np.float64}
