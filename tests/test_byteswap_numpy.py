# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

from awkward._nplikes.numpy import Numpy

DTYPES = [
    np.dtype(np.int8),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.int64),
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.uint32),
    np.dtype(np.uint64),
    np.dtype(np.float32),
    np.dtype(np.float64),
    np.dtype(np.complex64),
    np.dtype(np.complex128),
    np.dtype("datetime64[ns]"),
    np.dtype("timedelta64[ns]"),
]

SHAPES = [(12,), (3, 4), (2, 3, 4)]
LAYOUTS = ["contiguous", "sliced", "transposed"]


def _base_array(dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    size = int(np.prod(shape))

    if dtype.kind == "M":
        return np.arange(size, dtype=np.int64).astype(dtype).reshape(shape)
    if dtype.kind == "m":
        return np.arange(size, dtype=np.int64).astype(dtype).reshape(shape)
    if dtype.kind == "c":
        real = np.arange(size, dtype=np.float64)
        imag = real + 0.5
        return (real + 1j * imag).astype(dtype).reshape(shape)

    return np.arange(size, dtype=np.int64).astype(dtype).reshape(shape)


def _layout(base: np.ndarray, layout: str) -> np.ndarray:
    if layout == "contiguous":
        return np.ascontiguousarray(base)
    if layout == "sliced":
        return base[..., ::2] if base.ndim > 1 else base[::2]
    if layout == "transposed":
        if base.ndim < 2:
            return base
        if base.ndim == 2:
            return base.T
        return np.swapaxes(base, 0, 2)
    raise AssertionError(f"unknown layout: {layout}")


@pytest.mark.parametrize("dtype", DTYPES, ids=[str(x) for x in DTYPES])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout_name", LAYOUTS)
def test_numpy_byteswap_matches_numpy_reference(
    dtype: np.dtype, shape: tuple[int, ...], layout_name: str
):
    base = _base_array(dtype, shape)
    array = _layout(base, layout_name)

    nplike = Numpy.instance()
    backend_array = nplike.asarray(array)

    swapped = nplike.byteswap(backend_array)
    expected = array.byteswap()
    got = np.asarray(swapped)

    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    assert got.tobytes() == expected.tobytes()

    # Double byteswap must be identity at byte level.
    roundtrip = nplike.byteswap(swapped)
    rt = np.asarray(roundtrip)

    assert rt.shape == array.shape
    assert rt.dtype == array.dtype
    assert rt.tobytes() == array.tobytes()
