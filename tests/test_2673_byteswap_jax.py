# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import warnings

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.jax import Jax
from awkward._nplikes.virtual import VirtualNDArray

DTYPES = [
    np.dtype(np.int8),
    np.dtype(np.int32),
    np.dtype(np.int64),
    np.dtype(np.float32),
    np.dtype(np.float64),
    np.dtype(np.complex64),
]

SHAPES = [(6,), (2, 3)]
LAYOUTS = ["contiguous", "transposed", "zero_length"]


def _jax_nplike() -> Jax:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    ak.jax.register_and_check()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return Jax.instance()


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
def test_jax_byteswap_matches_numpy_reference(
    dtype: np.dtype, shape: tuple[int, ...], layout_name: str
):
    nplike = _jax_nplike()
    array = _layout(_base_array(dtype, shape), layout_name)

    try:
        backend_array = nplike.asarray(array)
    except (TypeError, ValueError, NotImplementedError):
        pytest.skip(f"jax does not support dtype {dtype}")

    swapped = nplike.byteswap(backend_array)
    expected = array.byteswap()
    got = np.asarray(swapped)

    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)

    roundtrip = np.asarray(nplike.byteswap(swapped))
    assert roundtrip.shape == array.shape
    assert roundtrip.dtype == array.dtype
    np.testing.assert_array_equal(roundtrip, array)


def test_jax_byteswap_virtualarray_is_lazy_and_remains_virtual(monkeypatch):
    nplike = _jax_nplike()
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
        buffer_key="node1-data",
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
    np.testing.assert_array_equal(np.asarray(swapped.materialize()), expected)

    assert swapped.is_materialized
    assert counts == {"generator": 1, "byteswap": 2}


def test_jax_byteswap_datetime64_raises():
    nplike = _jax_nplike()
    array = np.array([1, 2, 3], dtype="datetime64[ns]")

    with pytest.raises(TypeError, match="not a valid JAX array type"):
        nplike.asarray(array)
