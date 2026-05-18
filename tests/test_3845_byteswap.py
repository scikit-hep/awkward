# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

from awkward._nplikes.numpy import Numpy

nplike = Numpy.instance()


def _byteswap(arr):
    dtype = arr.dtype
    original_shape = arr.shape

    if np.issubdtype(dtype, np.complexfloating):
        component_dtype = np.finfo(dtype).dtype
        float_view = np.ascontiguousarray(arr).view(component_dtype)
        swapped = _byteswap(float_view)
        return np.ascontiguousarray(swapped).view(dtype).reshape(original_shape)

    itemsize = dtype.itemsize

    if itemsize == 1:
        return arr.copy()

    bytes_arr = np.ascontiguousarray(arr).view(np.uint8)
    bytes_arr = bytes_arr.reshape(-1, itemsize)
    bytes_arr = bytes_arr[..., ::-1]
    return (
        np.ascontiguousarray(bytes_arr.reshape(-1)).view(dtype).reshape(original_shape)
    )


SHAPES = [
    (),
    (1,),
    (9,),
    (1, 1),
    (1, 9),
    (9, 1),
    (3, 3),
    (1, 1, 1),
    (2, 3, 4),
    (1, 3, 1),
    (0,),
    (0, 0),
    (0, 3),
    (3, 0),
    (1, 0),
    (0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (0, 0, 0),
    (2, 0, 4),
    (1, 0, 5, 1),
]


def _check(values, shape):
    arr = np.resize(values, shape).astype(values.dtype, copy=False)
    expected = arr.byteswap()
    result = _byteswap(arr)
    assert result.dtype == arr.dtype
    assert result.shape == arr.shape
    np.testing.assert_array_equal(result, expected)

    nplike_result = nplike.byteswap(arr)
    assert nplike_result.dtype == arr.dtype
    assert nplike_result.shape == arr.shape
    np.testing.assert_array_equal(nplike_result, expected)


@pytest.mark.parametrize("shape", SHAPES)
def test_bool(shape):
    values = np.array([True, False, True, False, True], dtype=np.bool_)
    _check(values, shape)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_signed_integers(dtype, shape):
    info = np.iinfo(dtype)
    values = np.array(
        [0, 1, -1, info.min, info.max, info.min + 1, info.max - 1, 42, -42],
        dtype=dtype,
    )
    _check(values, shape)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
def test_unsigned_integers(dtype, shape):
    info = np.iinfo(dtype)
    values = np.array(
        [0, 1, info.max, info.max - 1, info.max // 2, 42, 255],
        dtype=dtype,
    )
    _check(values, shape)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_floats(dtype, shape):
    info = np.finfo(dtype)
    values = np.array(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            np.inf,
            -np.inf,
            np.nan,
            info.max,
            info.min,
            info.tiny,
            info.eps,
            3.14159265358979,
            -2.71828182845905,
        ],
        dtype=dtype,
    )
    _check(values, shape)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex(dtype, shape):
    component = np.finfo(dtype).dtype
    info = np.finfo(component)
    values = np.array(
        [
            0 + 0j,
            1 + 0j,
            0 + 1j,
            -1 - 1j,
            complex(np.inf, -np.inf),
            complex(np.nan, np.nan),
            complex(info.max, info.min),
            complex(info.tiny, info.eps),
            complex(3.14159, -2.71828),
        ],
        dtype=dtype,
    )
    _check(values, shape)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "m", "h", "D", "W", "M", "Y"])
def test_datetime64(unit, shape):
    values = np.array(
        ["1970-01-01", "2020-01-01", "2021-06-15", "2099-12-31", "1900-01-01"],
        dtype=f"datetime64[{unit}]",
    )
    _check(values, shape)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "m", "h", "D", "W"])
def test_timedelta64(unit, shape):
    info = np.iinfo(np.int64)
    values = np.array(
        [0, 1, -1, 1000, -1000, 86400, info.max, info.min + 1],
        dtype=f"timedelta64[{unit}]",
    )
    _check(values, shape)


def test_placeholder():
    from awkward._nplikes.placeholder import PlaceholderArray

    ph = PlaceholderArray(nplike, shape=(10,), dtype=np.dtype(np.int32))
    result = nplike.byteswap(ph)
    assert result is ph


def test_virtual():
    from awkward._nplikes.virtual import VirtualNDArray

    arr = np.array(
        [0, 1, -1, np.iinfo(np.int32).min, np.iinfo(np.int32).max], dtype=np.int32
    )
    expected = arr.byteswap()

    v = VirtualNDArray(nplike, shape=arr.shape, dtype=arr.dtype, generator=lambda: arr)
    assert not v.is_materialized
    result = nplike.byteswap(v)
    assert isinstance(result, VirtualNDArray)
    assert result.dtype == arr.dtype
    assert result.shape == arr.shape
    np.testing.assert_array_equal(result.materialize(), expected)

    v2 = VirtualNDArray(nplike, shape=arr.shape, dtype=arr.dtype, generator=lambda: arr)
    v2.materialize()
    result2 = nplike.byteswap(v2)
    np.testing.assert_array_equal(np.asarray(result2), expected)


def test_typetracer():
    from awkward._nplikes.typetracer import TypeTracer, TypeTracerArray

    tt = TypeTracer.instance()
    x = TypeTracerArray._new(np.dtype(np.float64), shape=(7,))
    result = tt.byteswap(x)
    assert isinstance(result, TypeTracerArray)
    assert result.dtype == x.dtype
    assert result.shape == x.shape
