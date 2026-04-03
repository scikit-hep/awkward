# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

from awkward._nplikes.cupy import Cupy
from awkward._nplikes.virtual import VirtualNDArray

DTYPES = [
    np.dtype(np.int8),
    np.dtype(np.int32),
    np.dtype(np.float16),
    np.dtype(np.complex64),
]

SHAPES = [(6,), (2, 3)]
LAYOUTS = ["contiguous", "transposed", "zero_length"]


def _cupy_nplike():
    cp = pytest.importorskip("cupy")

    return cp, Cupy.instance()


def _numpy_backed_cupy_nplike() -> Cupy:
    nplike = object.__new__(Cupy)
    nplike._module = np
    return nplike


def _base_array(dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    size = int(np.prod(shape))

    if dtype.kind == "c":
        real = np.arange(size, dtype=np.float32)
        imag = real + 0.5
        return (real + 1j * imag).astype(dtype).reshape(shape)
    return np.arange(size, dtype=np.int64).astype(dtype).reshape(shape)


def _layout(base: np.ndarray, layout_name: str) -> np.ndarray:
    if layout_name == "contiguous":
        return np.ascontiguousarray(base)
    if layout_name == "transposed":
        return base if base.ndim < 2 else base.T
    if layout_name == "zero_length":
        return base.reshape(-1)[:0]
    raise AssertionError(f"unknown layout: {layout_name}")


@pytest.mark.parametrize("dtype", DTYPES, ids=[str(x) for x in DTYPES])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout_name", LAYOUTS)
def test_cupy_byteswap_matches_numpy_reference(
    dtype: np.dtype, shape: tuple[int, ...], layout_name: str
):
    _cp, nplike = _cupy_nplike()
    array = _layout(_base_array(dtype, shape), layout_name)

    try:
        backend_array = nplike.asarray(array)
    except (TypeError, ValueError, NotImplementedError):
        pytest.skip(f"cupy does not support dtype {dtype}")

    swapped = nplike.byteswap(backend_array)
    expected = array.byteswap()
    got = swapped.get()

    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)

    roundtrip = nplike.byteswap(swapped).get()
    assert roundtrip.shape == array.shape
    assert roundtrip.dtype == array.dtype
    np.testing.assert_array_equal(roundtrip, array)


def test_cupy_byteswap_virtualarray_is_lazy_and_remains_virtual(monkeypatch):
    cp, nplike = _cupy_nplike()
    counts = {"generator": 0, "byteswap": 0}

    original_byteswap = nplike.byteswap

    def tracked_byteswap(array):
        counts["byteswap"] += 1
        return original_byteswap(array)

    monkeypatch.setattr(nplike, "byteswap", tracked_byteswap)

    def generator():
        counts["generator"] += 1
        return cp.array([1, 2, 3], dtype=cp.int32)

    virtual = VirtualNDArray(
        nplike,
        shape=(3,),
        dtype=np.dtype(np.int32),
        generator=generator,
        buffer_key="node2-data",
    )

    swapped = nplike.byteswap(virtual)

    assert isinstance(swapped, VirtualNDArray)
    assert not virtual.is_materialized
    assert not swapped.is_materialized
    assert swapped.buffer_key == virtual.buffer_key
    assert swapped.dtype == virtual.dtype
    assert swapped.shape == virtual.shape
    assert counts == {"generator": 0, "byteswap": 1}

    expected = np.array([1, 2, 3], dtype=np.int32).byteswap()
    np.testing.assert_array_equal(swapped.materialize().get(), expected)

    assert swapped.is_materialized
    assert counts == {"generator": 1, "byteswap": 2}


def test_cupy_byteswap_int8_copy_path_without_cupy():
    nplike = _numpy_backed_cupy_nplike()
    array = np.array([1, 2, 3], dtype=np.int8)

    swapped = nplike.byteswap(array)

    np.testing.assert_array_equal(swapped, array.byteswap())
    assert swapped is not array


def test_cupy_byteswap_float16_without_cupy():
    nplike = _numpy_backed_cupy_nplike()
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16).T

    swapped = nplike.byteswap(array)

    np.testing.assert_array_equal(swapped, array.byteswap())


def test_cupy_byteswap_complex64_without_cupy():
    nplike = _numpy_backed_cupy_nplike()
    array = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)

    swapped = nplike.byteswap(array)

    np.testing.assert_array_equal(swapped, array.byteswap())


def test_cupy_byteswap_virtualarray_is_lazy_without_cupy(monkeypatch):
    nplike = _numpy_backed_cupy_nplike()
    counts = {"generator": 0, "byteswap": 0}

    original_byteswap = nplike.byteswap

    def tracked_byteswap(array):
        counts["byteswap"] += 1
        return original_byteswap(array)

    monkeypatch.setattr(nplike, "byteswap", tracked_byteswap)

    def generator():
        counts["generator"] += 1
        return np.array([1, 2, 3], dtype=np.int32)

    virtual = VirtualNDArray(
        nplike,
        shape=(3,),
        dtype=np.dtype(np.int32),
        generator=generator,
        buffer_key="node3-data",
    )

    swapped = nplike.byteswap(virtual)

    assert isinstance(swapped, VirtualNDArray)
    assert not virtual.is_materialized
    assert not swapped.is_materialized
    assert counts == {"generator": 0, "byteswap": 1}
    expected = np.array([1, 2, 3], dtype=np.int32).byteswap()
    np.testing.assert_array_equal(swapped.materialize(), expected)
    assert counts == {"generator": 1, "byteswap": 2}
