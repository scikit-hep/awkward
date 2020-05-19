# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_typeempty():
    assert numpy.asarray(awkward1.types.PrimitiveType("bool").empty()).dtype == numpy.dtype("bool")
    assert numpy.asarray(awkward1.types.PrimitiveType("int8").empty()).dtype == numpy.dtype("int8")
    assert numpy.asarray(awkward1.types.PrimitiveType("int16").empty()).dtype == numpy.dtype("int16")
    assert numpy.asarray(awkward1.types.PrimitiveType("int32").empty()).dtype == numpy.dtype("int32")
    assert numpy.asarray(awkward1.types.PrimitiveType("int64").empty()).dtype == numpy.dtype("int64")
    assert numpy.asarray(awkward1.types.PrimitiveType("uint8").empty()).dtype == numpy.dtype("uint8")
    assert numpy.asarray(awkward1.types.PrimitiveType("uint16").empty()).dtype == numpy.dtype("uint16")
    assert numpy.asarray(awkward1.types.PrimitiveType("uint32").empty()).dtype == numpy.dtype("uint32")
    assert numpy.asarray(awkward1.types.PrimitiveType("uint64").empty()).dtype == numpy.dtype("uint64")
    assert numpy.asarray(awkward1.types.PrimitiveType("float32").empty()).dtype == numpy.dtype("float32")
    assert numpy.asarray(awkward1.types.PrimitiveType("float64").empty()).dtype == numpy.dtype("float64")
    assert type(awkward1.types.UnknownType().empty()) is awkward1.layout.EmptyArray
    assert type(awkward1.types.ArrayType(awkward1.types.UnknownType(), 0).empty()) is awkward1.layout.EmptyArray
    assert type(awkward1.types.RegularType(awkward1.types.UnknownType(), 5).empty()) is awkward1.layout.RegularArray
    assert type(awkward1.types.ListType(awkward1.types.UnknownType()).empty()) is awkward1.layout.ListOffsetArray64
    array = awkward1.types.RecordType({"one": awkward1.types.PrimitiveType("float64"), "two": awkward1.types.ListType(awkward1.types.PrimitiveType("float64"))}).empty()
    assert type(array) is awkward1.layout.RecordArray
    assert type(array["one"]) is awkward1.layout.NumpyArray
    assert numpy.asarray(array["one"]).dtype == numpy.dtype("float64")
    assert type(array["two"]) is awkward1.layout.ListOffsetArray64
