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
