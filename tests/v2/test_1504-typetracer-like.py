import awkward as ak
import numpy as np
import pytest


typetracer = ak._v2._typetracer.TypeTracer.instance()


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint8, None])
@pytest.mark.parametrize("like_dtype", [np.float64, np.int64, np.uint8, None])
def test_ones_like(dtype, like_dtype):
    array = ak._v2.contents.numpyarray.NumpyArray(
        np.array([99, 88, 77, 66, 66], dtype=dtype)
    )
    ones = ak._v2.ones_like(array.typetracer, dtype=like_dtype, highlevel=False)
    assert ones.typetracer.shape == array.shape
    assert ones.typetracer.dtype == like_dtype or array.dtype


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint8, None])
@pytest.mark.parametrize("like_dtype", [np.float64, np.int64, np.uint8, None])
def test_zeros_like(dtype, like_dtype):
    array = ak._v2.contents.numpyarray.NumpyArray(
        np.array([99, 88, 77, 66, 66], dtype=dtype)
    )

    full = ak._v2.zeros_like(array.typetracer, dtype=like_dtype, highlevel=False)
    assert full.typetracer.shape == array.shape
    assert full.typetracer.dtype == like_dtype or array.dtype


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint8, None])
@pytest.mark.parametrize("like_dtype", [np.float64, np.int64, np.uint8, None])
@pytest.mark.parametrize("value", [1.0, -20, np.iinfo(np.uint64).max])
def test_full_like(dtype, like_dtype, value):
    array = ak._v2.contents.numpyarray.NumpyArray(
        np.array([99, 88, 77, 66, 66], dtype=dtype)
    )
    full = ak._v2.full_like(array.typetracer, value, dtype=like_dtype, highlevel=False)
    assert full.typetracer.shape == array.shape
    assert full.typetracer.dtype == like_dtype or array.dtype


def test_full_like_cast():
    with pytest.raises(ValueError):
        ak._v2._typetracer.TypeTracerArray.from_array(
            [1, ak._v2._typetracer.UnknownScalar(np.uint8)]
        )
