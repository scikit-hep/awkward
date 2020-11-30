# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak


def test_typeempty():
    assert np.asarray(ak.types.PrimitiveType("bool").empty()).dtype == np.dtype("bool")
    assert np.asarray(ak.types.PrimitiveType("int8").empty()).dtype == np.dtype("int8")
    assert np.asarray(ak.types.PrimitiveType("int16").empty()).dtype == np.dtype(
        "int16"
    )
    assert np.asarray(ak.types.PrimitiveType("int32").empty()).dtype == np.dtype(
        "int32"
    )
    assert np.asarray(ak.types.PrimitiveType("int64").empty()).dtype == np.dtype(
        "int64"
    )
    assert np.asarray(ak.types.PrimitiveType("uint8").empty()).dtype == np.dtype(
        "uint8"
    )
    assert np.asarray(ak.types.PrimitiveType("uint16").empty()).dtype == np.dtype(
        "uint16"
    )
    assert np.asarray(ak.types.PrimitiveType("uint32").empty()).dtype == np.dtype(
        "uint32"
    )
    assert np.asarray(ak.types.PrimitiveType("uint64").empty()).dtype == np.dtype(
        "uint64"
    )
    assert np.asarray(ak.types.PrimitiveType("float32").empty()).dtype == np.dtype(
        "float32"
    )
    assert np.asarray(ak.types.PrimitiveType("float64").empty()).dtype == np.dtype(
        "float64"
    )
    assert type(ak.types.UnknownType().empty()) is ak.layout.EmptyArray
    assert (
        type(ak.types.ArrayType(ak.types.UnknownType(), 0).empty())
        is ak.layout.EmptyArray
    )
    assert (
        type(ak.types.RegularType(ak.types.UnknownType(), 5).empty())
        is ak.layout.RegularArray
    )
    assert (
        type(ak.types.ListType(ak.types.UnknownType()).empty())
        is ak.layout.ListOffsetArray64
    )
    array = ak.types.RecordType(
        {
            "one": ak.types.PrimitiveType("float64"),
            "two": ak.types.ListType(ak.types.PrimitiveType("float64")),
        }
    ).empty()
    assert type(array) is ak.layout.RecordArray
    assert type(array["one"]) is ak.layout.NumpyArray
    assert np.asarray(array["one"]).dtype == np.dtype("float64")
    assert type(array["two"]) is ak.layout.ListOffsetArray64
