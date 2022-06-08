# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test_UnmaskedArray():
    content_float64 = ak._v2.contents.NumpyArray(
        np.array([0.25, 0.5, 3.5, 4.5, 5.5], dtype=np.float64)
    )
    array_float64 = ak._v2.contents.UnmaskedArray(content_float64)
    assert to_list(array_float64) == [0.25, 0.5, 3.5, 4.5, 5.5]
    assert str(ak._v2.operations.type(content_float64)) == "float64"
    assert (
        str(ak._v2.operations.type(ak._v2.highlevel.Array(content_float64)))
        == "5 * float64"
    )
    assert str(ak._v2.operations.type(array_float64)) == "?float64"
    assert (
        str(ak._v2.operations.type(ak._v2.highlevel.Array(array_float64)))
        == "5 * ?float64"
    )

    assert np.can_cast(np.float32, np.float64) is True
    assert np.can_cast(np.float64, np.float32, "unsafe") is True
    assert np.can_cast(np.float64, np.int8, "unsafe") is True

    content_float32 = ak._v2.operations.values_astype(
        content_float64, "float32", highlevel=False
    )
    array_float32 = ak._v2.contents.UnmaskedArray(content_float32)
    assert to_list(array_float32) == [0.25, 0.5, 3.5, 4.5, 5.5]
    assert str(ak._v2.operations.type(content_float32)) == "float32"
    assert (
        str(ak._v2.operations.type(ak._v2.highlevel.Array(content_float32)))
        == "5 * float32"
    )
    assert str(ak._v2.operations.type(array_float32)) == "?float32"
    assert (
        str(ak._v2.operations.type(ak._v2.highlevel.Array(array_float32)))
        == "5 * ?float32"
    )

    content_int8 = ak._v2.operations.values_astype(
        content_float64, "int8", highlevel=False
    )
    array_int8 = ak._v2.contents.UnmaskedArray(content_int8)
    assert to_list(array_int8) == [0, 0, 3, 4, 5]
    assert str(ak._v2.operations.type(content_int8)) == "int8"
    assert (
        str(ak._v2.operations.type(ak._v2.highlevel.Array(content_int8))) == "5 * int8"
    )
    assert str(ak._v2.operations.type(array_int8)) == "?int8"
    assert (
        str(ak._v2.operations.type(ak._v2.highlevel.Array(array_int8))) == "5 * ?int8"
    )

    content_from_int8 = ak._v2.operations.values_astype(
        content_int8, "float64", highlevel=False
    )
    array_from_int8 = ak._v2.contents.UnmaskedArray(content_from_int8)
    assert to_list(array_from_int8) == [0, 0, 3, 4, 5]
    assert str(ak._v2.operations.type(content_from_int8)) == "float64"
    assert (
        str(ak._v2.operations.type(ak._v2.highlevel.Array(content_from_int8)))
        == "5 * float64"
    )
    assert str(ak._v2.operations.type(array_from_int8)) == "?float64"
    assert (
        str(ak._v2.operations.type(ak._v2.highlevel.Array(array_from_int8)))
        == "5 * ?float64"
    )


def test_RegularArray_and_ListArray():
    content = ak._v2.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak._v2.contents.ListOffsetArray(offsets, content)
    regulararray = ak._v2.contents.RegularArray(listoffsetarray, 2, zeros_length=0)
    starts = ak._v2.index.Index64(np.array([0, 1]))
    stops = ak._v2.index.Index64(np.array([2, 3]))
    listarray = ak._v2.contents.ListArray(starts, stops, regulararray)

    assert str(ak._v2.operations.type(content)) == "float64"
    assert str(ak._v2.operations.type(regulararray)) == "2 * var * float64"
    assert str(ak._v2.operations.type(listarray)) == "var * 2 * var * float64"

    regulararray_int8 = ak._v2.operations.values_astype(
        regulararray, "int8", highlevel=False
    )
    assert str(ak._v2.operations.type(regulararray_int8)) == "2 * var * int8"

    listarray_bool = ak._v2.operations.values_astype(listarray, "bool", highlevel=False)
    assert str(ak._v2.operations.type(listarray_bool)) == "var * 2 * var * bool"


def test_ufunc_afterward():
    assert (
        ak._v2.operations.values_astype(
            ak._v2.highlevel.Array([{"x": 1.1}, {"x": 3.3}]), np.float32
        )["x"]
        + 1
    ).tolist() == [2.0999999046325684, 4.300000190734863]


def test_string():
    assert to_list(
        ak._v2.operations.values_astype(
            ak._v2.highlevel.Array([{"x": 1.1, "y": "hello"}]), np.float32
        )
    ) == [{"x": 1.100000023841858, "y": "hello"}]
