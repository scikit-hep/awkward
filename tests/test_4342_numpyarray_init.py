# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward.types.numpytype import (
    _primitive_to_dtype_dict,
    dtype_to_primitive,
    primitive_to_dtype,
)


def test_scalar_is_rejected():
    with pytest.raises(TypeError, match="must be an array, not a scalar"):
        ak.contents.NumpyArray(np.array(1.1))


@pytest.mark.parametrize("shape", [(6,), (2, 3), (1, 2, 3)])
def test_shape_is_preserved(shape):
    layout = ak.contents.NumpyArray(np.arange(6, dtype=np.int64).reshape(shape))
    assert layout.shape == shape
    assert layout.inner_shape == shape[1:]
    assert layout.dtype == np.dtype(np.int64)


@pytest.mark.parametrize("primitive", sorted(_primitive_to_dtype_dict))
def test_primitive_round_trip(primitive):
    dtype = primitive_to_dtype(primitive)
    assert dtype_to_primitive(dtype) == primitive
    layout = ak.contents.NumpyArray(np.empty(3, dtype=dtype))
    assert str(ak.type(layout)) == f"3 * {primitive}"


@pytest.mark.parametrize("dtype", [np.dtype(object), np.dtype(">f8")])
def test_unsupported_dtype(dtype):
    with pytest.raises(TypeError, match="unsupported dtype"):
        dtype_to_primitive(dtype)
    with pytest.raises(TypeError, match="unsupported dtype"):
        ak.contents.NumpyArray(np.empty(3, dtype=dtype))


@pytest.mark.parametrize(
    "primitive",
    ["datetime64[ns]", "datetime64[D]", "timedelta64[us]", "timedelta64[s]"],
)
def test_datetime_and_timedelta_units(primitive):
    # `dtype.kind in "mM"` must accept both "M" (datetime64) and "m" (timedelta64)
    dtype = np.dtype(primitive)
    assert dtype_to_primitive(dtype) == primitive
    layout = ak.contents.NumpyArray(np.empty(3, dtype=dtype))
    assert str(ak.type(layout)) == f"3 * {primitive}"


@pytest.mark.parametrize("primitive", [">M8[ns]", ">m8[us]"])
def test_non_native_byteorder_is_rejected(primitive):
    dtype = np.dtype(primitive)
    assert dtype.kind in "mM"
    with pytest.raises(TypeError, match="unsupported dtype"):
        dtype_to_primitive(dtype)
    with pytest.raises(TypeError, match="unsupported dtype"):
        ak.contents.NumpyArray(np.empty(3, dtype=dtype))


@pytest.mark.parametrize("dtype", ["float32", "datetime64[ns]", "timedelta64[us]"])
def test_typetracer_form_matches(dtype):
    array = np.empty(3, dtype=dtype)
    layout = ak.contents.NumpyArray(array)
    tt = ak.Array(layout).layout.to_typetracer(forget_length=True)
    assert tt.backend.name == "typetracer"
    assert tt.form == layout.form
