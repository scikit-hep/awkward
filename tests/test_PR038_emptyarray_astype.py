# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_typeempty():
    assert numpy.asarray(awkward1.layout.PrimitiveType("bool").empty()).dtype == numpy.dtype("bool")
    assert numpy.asarray(awkward1.layout.PrimitiveType("int8").empty()).dtype == numpy.dtype("int8")
    assert numpy.asarray(awkward1.layout.PrimitiveType("int16").empty()).dtype == numpy.dtype("int16")
    assert numpy.asarray(awkward1.layout.PrimitiveType("int32").empty()).dtype == numpy.dtype("int32")
    assert numpy.asarray(awkward1.layout.PrimitiveType("int64").empty()).dtype == numpy.dtype("int64")
    assert numpy.asarray(awkward1.layout.PrimitiveType("uint8").empty()).dtype == numpy.dtype("uint8")
    assert numpy.asarray(awkward1.layout.PrimitiveType("uint16").empty()).dtype == numpy.dtype("uint16")
    assert numpy.asarray(awkward1.layout.PrimitiveType("uint32").empty()).dtype == numpy.dtype("uint32")
    assert numpy.asarray(awkward1.layout.PrimitiveType("uint64").empty()).dtype == numpy.dtype("uint64")
    assert numpy.asarray(awkward1.layout.PrimitiveType("float32").empty()).dtype == numpy.dtype("float32")
    assert numpy.asarray(awkward1.layout.PrimitiveType("float64").empty()).dtype == numpy.dtype("float64")
    assert type(awkward1.layout.UnknownType().empty()) is awkward1.layout.EmptyArray
    assert type(awkward1.layout.ArrayType(awkward1.layout.UnknownType(), 0).empty()) is awkward1.layout.EmptyArray
    assert type(awkward1.layout.RegularType(awkward1.layout.UnknownType(), 5).empty()) is awkward1.layout.RegularArray
    assert type(awkward1.layout.ListType(awkward1.layout.UnknownType()).empty()) is awkward1.layout.ListOffsetArray64
    array = awkward1.layout.RecordType({"one": awkward1.layout.PrimitiveType("float64"), "two": awkward1.layout.ListType(awkward1.layout.PrimitiveType("float64"))}).empty()
    assert type(array) is awkward1.layout.RecordArray
    assert type(array["one"]) is awkward1.layout.NumpyArray
    assert numpy.asarray(array["one"]).dtype == numpy.dtype("float64")
    assert type(array["two"]) is awkward1.layout.ListOffsetArray64
