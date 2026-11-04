# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

cp = pytest.importorskip("cupy")
pytest.importorskip("numba_cuda_mlir")


@pytest.mark.parametrize("validwhen", [False, True])
@pytest.mark.parametrize("lsb_order", [False, True])
@pytest.mark.parametrize("length", [0, 1, 65, 256])
def test_bitmask_cpu_parity(validwhen, lsb_order, length):
    from awkward._backends.cupy import CupyBackend
    from awkward._backends.numpy import NumpyBackend

    source = np.arange(256, dtype=np.uint8)[:length]
    for name, dtype in [
        ("awkward_BitMaskedArray_to_ByteMaskedArray", np.int8),
        ("awkward_BitMaskedArray_to_IndexedOptionArray", np.int64),
    ]:
        expected = np.full(length * 8 + 2, -99, dtype=dtype)
        actual = cp.asarray(expected)
        key = (name, dtype, np.uint8)
        error = NumpyBackend.instance()[key](
            expected, source, length, validwhen, lsb_order
        )
        assert not error.str
        CupyBackend.instance()[key](
            actual, cp.asarray(source), length, validwhen, lsb_order
        )
        ak._connect.cuda.synchronize_cuda()
        np.testing.assert_array_equal(cp.asnumpy(actual), expected)


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
    ],
)
@pytest.mark.parametrize("operation", [ak.argmin, ak.argmax], ids=["argmin", "argmax"])
def test_numeric_argminmax_cpu_parity(dtype, operation):
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        small = np.array([np.nan, 1, 2, 1, np.nan, -1, -np.inf, np.inf, 0], dtype=dtype)
        low, high = -3, 3
    else:
        low, high = (
            (False, True)
            if dtype.kind == "b"
            else (np.iinfo(dtype).min, np.iinfo(dtype).max)
        )
        small = np.array([high, low, high, low, high, low, high, 0, 0], dtype=dtype)
    large = np.zeros(513, dtype=dtype)
    large[255], large[512] = high, low
    cpu = ak.unflatten(np.concatenate([small, large]), [0, 3, 4, 2, 513])
    gpu = ak.to_backend(cpu, "cuda")
    expected = operation(cpu, axis=-1, mask_identity=False)
    actual = operation(gpu, axis=-1, mask_identity=False)
    assert ak.backend(actual) == "cuda"
    assert ak.to_list(actual) == ak.to_list(expected)
    ak._connect.cuda.synchronize_cuda()
