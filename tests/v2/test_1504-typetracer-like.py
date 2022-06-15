import awkward as ak
import numpy as np
import pytest


typetracer = ak._v2._typetracer.TypeTracer.instance()


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint8, None])
@pytest.mark.parametrize("like_dtype", [np.float64, np.int64, np.uint8, None])
def test(dtype, like_dtype):
    a = ak._v2.contents.numpyarray.NumpyArray(
        np.array([99, 88, 77, 66, 66], dtype=dtype)
    )
    ones = ak._v2.ones_like(a.typetracer, dtype=like_dtype, highlevel=False)
    assert ones.typetracer.shape == a.shape
    assert ones.typetracer.dtype == like_dtype or a.dtype


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint8, None])
@pytest.mark.parametrize("like_dtype", [np.float64, np.int64, np.uint8, None])
def test(dtype, like_dtype):
    a = ak._v2.contents.numpyarray.NumpyArray(
        np.array([99, 88, 77, 66, 66], dtype=dtype)
    )

    full = ak._v2.zeros_like(a.typetracer, dtype=like_dtype, highlevel=False)
    assert full.typetracer.shape == a.shape
    assert full.typetracer.dtype == like_dtype or a.dtype


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint8, None])
@pytest.mark.parametrize("like_dtype", [np.float64, np.int64, np.uint8, None])
@pytest.mark.parametrize("value", [1.0, -20, np.iinfo(np.uint64).max])
def test(dtype, like_dtype, value):
    a = ak._v2.contents.numpyarray.NumpyArray(
        np.array([99, 88, 77, 66, 66], dtype=dtype)
    )
    full = ak._v2.full_like(a.typetracer, value, dtype=like_dtype, highlevel=False)
    assert full.typetracer.shape == a.shape
    assert full.typetracer.dtype == like_dtype or a.dtype
