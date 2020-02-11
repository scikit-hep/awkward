# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_typeempty():
    assert numpy.asarray(awkward1.type.PrimitiveType("bool").empty()).dtype == numpy.dtype("bool")
    assert numpy.asarray(awkward1.type.PrimitiveType("int8").empty()).dtype == numpy.dtype("int8")
    assert numpy.asarray(awkward1.type.PrimitiveType("int16").empty()).dtype == numpy.dtype("int16")
    assert numpy.asarray(awkward1.type.PrimitiveType("int32").empty()).dtype == numpy.dtype("int32")
    assert numpy.asarray(awkward1.type.PrimitiveType("int64").empty()).dtype == numpy.dtype("int64")
    assert numpy.asarray(awkward1.type.PrimitiveType("uint8").empty()).dtype == numpy.dtype("uint8")
    assert numpy.asarray(awkward1.type.PrimitiveType("uint16").empty()).dtype == numpy.dtype("uint16")
    assert numpy.asarray(awkward1.type.PrimitiveType("uint32").empty()).dtype == numpy.dtype("uint32")
    assert numpy.asarray(awkward1.type.PrimitiveType("uint64").empty()).dtype == numpy.dtype("uint64")
    assert numpy.asarray(awkward1.type.PrimitiveType("float32").empty()).dtype == numpy.dtype("float32")
    assert numpy.asarray(awkward1.type.PrimitiveType("float64").empty()).dtype == numpy.dtype("float64")
    assert type(awkward1.type.UnknownType().empty()) is awkward1.layout.EmptyArray
    assert type(awkward1.type.ArrayType(awkward1.type.UnknownType(), 0).empty()) is awkward1.layout.EmptyArray
    assert type(awkward1.type.RegularType(awkward1.type.UnknownType(), 5).empty()) is awkward1.layout.RegularArray
    assert type(awkward1.type.ListType(awkward1.type.UnknownType()).empty()) is awkward1.layout.ListOffsetArray64
    array = awkward1.type.RecordType({"one": awkward1.type.PrimitiveType("float64"), "two": awkward1.type.ListType(awkward1.type.PrimitiveType("float64"))}).empty()
    assert type(array) is awkward1.layout.RecordArray
    assert type(array["one"]) is awkward1.layout.NumpyArray
    assert numpy.asarray(array["one"]).dtype == numpy.dtype("float64")
    assert type(array["two"]) is awkward1.layout.ListOffsetArray64

def test_astype():
    empty = awkward1.layout.EmptyArray()
    assert numpy.asarray(empty.astype(awkward1.type.PrimitiveType("bool"))).dtype == numpy.dtype("bool")
    assert numpy.asarray(empty.astype(awkward1.type.PrimitiveType("uint8"))).dtype == numpy.dtype("uint8")
    assert numpy.asarray(empty.astype(awkward1.type.PrimitiveType("float64"))).dtype == numpy.dtype("float64")
    assert type(empty.astype(awkward1.type.ListType(awkward1.type.PrimitiveType("float64")))) is awkward1.layout.ListOffsetArray64
