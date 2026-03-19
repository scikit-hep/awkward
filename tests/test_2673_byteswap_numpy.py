# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

from awkward._nplikes.array_module import ArrayModuleNumpyLike
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.virtual import VirtualNDArray

DTYPES = [
    np.dtype(np.int8),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.int64),
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.uint32),
    np.dtype(np.uint64),
    np.dtype(np.float16),
    np.dtype(np.float32),
    np.dtype(np.float64),
    np.dtype(np.complex64),
    np.dtype(np.complex128),
]

SHAPES = [(6,), (2, 3)]
LAYOUTS = ["contiguous", "sliced", "transposed", "zero_length"]
UNITS = ["ns", "us", "ms", "s", "m", "h", "D"]


def _repeat_values(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    size = int(np.prod(shape))
    return np.resize(values, size).reshape(shape)


def _base_array(dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    if dtype.kind == "c":
        values = np.array([1 + 2j, -3 + 4j, 0 + 0j, 5 - 6j], dtype=dtype)
    elif dtype.kind == "f":
        values = np.array([0.0, -0.0, 1.5, -2.5, np.inf, -np.inf], dtype=dtype)
    elif dtype.kind == "i":
        info = np.iinfo(dtype)
        values = np.array([0, 1, -1, info.max, info.min, 2], dtype=dtype)
    elif dtype.kind == "u":
        info = np.iinfo(dtype)
        max_value = int(info.max)
        values = np.array(
            [0, 1, 2, 3, max_value - 1 if max_value > 0 else max_value, max_value],
            dtype=dtype,
        )
    else:
        raise AssertionError(f"unexpected dtype kind: {dtype.kind}")

    return _repeat_values(values, shape)


def _layout(base: np.ndarray, layout_name: str) -> np.ndarray:
    if layout_name == "contiguous":
        return np.ascontiguousarray(base)
    if layout_name == "sliced":
        return base[::2] if base.ndim == 1 else base[:, ::2]
    if layout_name == "transposed":
        return base if base.ndim < 2 else base.T
    if layout_name == "zero_length":
        return base.reshape(-1)[:0]
    raise AssertionError(f"unknown layout: {layout_name}")


def test_array_module_byteswap_raises():
    class DummyArrayModuleNumpyLike(ArrayModuleNumpyLike):
        is_eager = True
        supports_structured_dtypes = True
        supports_virtual_arrays = False
        _module = np

        @property
        def ndarray(self):
            return np.ndarray

        @classmethod
        def is_own_array_type(cls, type_: type) -> bool:
            return issubclass(type_, np.ndarray)

        def is_c_contiguous(self, x) -> bool:
            return True

    base_nplike = object.__new__(DummyArrayModuleNumpyLike)

    with pytest.raises(NotImplementedError):
        base_nplike.byteswap(None)


@pytest.mark.parametrize("dtype", DTYPES, ids=[str(x) for x in DTYPES])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout_name", LAYOUTS)
def test_numpy_byteswap_matches_numpy_reference(
    dtype: np.dtype, shape: tuple[int, ...], layout_name: str
):
    array = _layout(_base_array(dtype, shape), layout_name)

    nplike = Numpy.instance()
    swapped = nplike.byteswap(nplike.asarray(array))
    expected = array.byteswap()
    got = np.asarray(swapped)

    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)
    assert got.tobytes() == expected.tobytes()

    roundtrip = np.asarray(nplike.byteswap(swapped))
    assert roundtrip.shape == array.shape
    assert roundtrip.dtype == array.dtype
    np.testing.assert_array_equal(roundtrip, array)
    assert roundtrip.tobytes() == array.tobytes()


def test_numpy_byteswap_virtualarray_is_lazy_and_remains_virtual(monkeypatch):
    nplike = Numpy.instance()
    counts = {"generator": 0, "byteswap": 0}

    original_byteswap = nplike.byteswap

    def tracked_byteswap(array):
        counts["byteswap"] += 1
        return original_byteswap(array)

    monkeypatch.setattr(nplike, "byteswap", tracked_byteswap)

    def generator():
        counts["generator"] += 1
        return np.array([[1, 2], [3, 4]], dtype=np.int32)

    virtual = VirtualNDArray(
        nplike,
        shape=(2, 2),
        dtype=np.dtype(np.int32),
        generator=generator,
        buffer_key="node0-data",
    )

    swapped = nplike.byteswap(virtual)

    assert isinstance(swapped, VirtualNDArray)
    assert not virtual.is_materialized
    assert not swapped.is_materialized
    assert swapped.buffer_key == virtual.buffer_key
    assert swapped.dtype == virtual.dtype
    assert swapped.shape == virtual.shape
    assert counts == {"generator": 0, "byteswap": 1}

    expected = np.array([[1, 2], [3, 4]], dtype=np.int32).byteswap()
    np.testing.assert_array_equal(np.asarray(swapped.materialize()), expected)

    assert swapped.is_materialized
    assert counts == {"generator": 1, "byteswap": 2}


def test_from_to_buffers_byteorder_roundtrip():
    import awkward as ak

    array = ak.Array([1, 2, 3, 4, 5])

    # big endian round-trip (">")
    form, length, buffers = ak.to_buffers(array, byteorder=">")
    result = ak.from_buffers(form, length, buffers, byteorder=">")
    assert ak.to_list(result) == [1, 2, 3, 4, 5]

    # little endian round-trip ("<")
    form, length, buffers = ak.to_buffers(array, byteorder="<")
    result = ak.from_buffers(form, length, buffers, byteorder="<")
    assert ak.to_list(result) == [1, 2, 3, 4, 5]


def test_from_to_buffers_byteorder_cross():
    import awkward as ak

    # export as big endian, reimport — values should still be correct
    array = ak.Array([10, 20, 30])
    form, length, buffers = ak.to_buffers(array, byteorder=">")
    result = ak.from_buffers(form, length, buffers, byteorder=">")
    assert ak.to_list(result) == [10, 20, 30]


@pytest.mark.parametrize("unit", UNITS)
def test_numpy_datetime64_byteswap_matches_numpy_reference(unit: str):
    dtype = np.dtype(f"datetime64[{unit}]")
    array = np.arange(6, dtype=np.int64).astype(dtype).reshape(2, 3)

    nplike = Numpy.instance()
    swapped = nplike.byteswap(nplike.asarray(array))
    expected = array.byteswap()
    got = np.asarray(swapped)

    assert got.dtype == dtype
    np.testing.assert_array_equal(got, expected)
    assert got.tobytes() == expected.tobytes()


@pytest.mark.parametrize("unit", UNITS)
def test_numpy_timedelta64_byteswap_matches_numpy_reference(unit: str):
    dtype = np.dtype(f"timedelta64[{unit}]")
    array = np.arange(6, dtype=np.int64).astype(dtype).reshape(2, 3)

    nplike = Numpy.instance()
    swapped = nplike.byteswap(nplike.asarray(array))
    expected = array.byteswap()
    got = np.asarray(swapped)

    assert got.dtype == dtype
    np.testing.assert_array_equal(got, expected)
    assert got.tobytes() == expected.tobytes()
