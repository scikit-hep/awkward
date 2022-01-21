# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_typeempty():
    assert np.asarray(ak._v2.types.NumpyType("bool").empty()).dtype == np.dtype("bool")
    assert np.asarray(ak._v2.types.NumpyType("int8").empty()).dtype == np.dtype("int8")
    assert np.asarray(ak._v2.types.NumpyType("int16").empty()).dtype == np.dtype(
        "int16"
    )
    assert np.asarray(ak._v2.types.NumpyType("int32").empty()).dtype == np.dtype(
        "int32"
    )
    assert np.asarray(ak._v2.types.NumpyType("int64").empty()).dtype == np.dtype(
        "int64"
    )
    assert np.asarray(ak._v2.types.NumpyType("uint8").empty()).dtype == np.dtype(
        "uint8"
    )
    assert np.asarray(ak._v2.types.NumpyType("uint16").empty()).dtype == np.dtype(
        "uint16"
    )
    assert np.asarray(ak._v2.types.NumpyType("uint32").empty()).dtype == np.dtype(
        "uint32"
    )
    assert np.asarray(ak._v2.types.NumpyType("uint64").empty()).dtype == np.dtype(
        "uint64"
    )
    assert np.asarray(ak._v2.types.NumpyType("float32").empty()).dtype == np.dtype(
        "float32"
    )
    assert np.asarray(ak._v2.types.NumpyType("float64").empty()).dtype == np.dtype(
        "float64"
    )
    assert type(ak._v2.types.UnknownType().empty()) is ak._v2.contents.EmptyArray
    assert (
        type(ak._v2.types.ArrayType(ak._v2.types.UnknownType(), 0).empty())
        is ak._v2.contents.emptyarray.EmptyArray
    )
    assert (
        type(ak._v2.types.RegularType(ak._v2.types.UnknownType(), 5).empty())
        is ak._v2.contents.RegularArray
    )
    assert (
        type(ak._v2.types.ListType(ak._v2.types.UnknownType()).empty())
        is ak._v2.contents.ListOffsetArray
    )


@pytest.mark.skip(reason="FIXME: Fields passing for RecordType v2 is different")
def test_recordtype():
    array = ak._v2.types.RecordType(
        {
            "one": ak._v2.types.NumpyType("float64"),
            "two": ak._v2.types.ListType(ak._v2.types.NumpyType("float64")),
        }
    ).empty()
    assert type(array) is ak._v2.contents.RecordArray
    assert type(array["one"]) is ak._v2.contents.NumpyArray
    assert np.asarray(array["one"]).dtype == np.dtype("float64")
    assert type(array["two"]) is ak._v2.contents.ListOffsetArray
