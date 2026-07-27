# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

from awkward._nplikes.cupy import Cupy

nplike = Cupy.instance()


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
    cp_arr = cp.resize(cp.asarray(values), shape).astype(values.dtype)
    expected = cp.asnumpy(cp_arr).byteswap()

    result = nplike.byteswap(cp_arr)
    assert result.dtype == values.dtype
    assert result.shape == shape
    np.testing.assert_array_equal(cp.asnumpy(result), expected)


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


def test_virtual():
    from awkward._nplikes.virtual import VirtualNDArray

    arr = cp.asarray(
        np.array(
            [0, 1, -1, np.iinfo(np.int32).min, np.iinfo(np.int32).max],
            dtype=np.int32,
        )
    )
    expected = cp.asnumpy(arr).byteswap()

    v = VirtualNDArray(nplike, shape=arr.shape, dtype=arr.dtype, generator=lambda: arr)
    assert not v.is_materialized
    result = nplike.byteswap(v)
    assert isinstance(result, VirtualNDArray)
    assert result.dtype == arr.dtype
    assert result.shape == arr.shape
    np.testing.assert_array_equal(cp.asnumpy(result.materialize()), expected)

    v2 = VirtualNDArray(nplike, shape=arr.shape, dtype=arr.dtype, generator=lambda: arr)
    v2.materialize()
    result2 = nplike.byteswap(v2)
    np.testing.assert_array_equal(cp.asnumpy(result2), expected)
