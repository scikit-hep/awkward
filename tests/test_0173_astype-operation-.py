# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_UnmaskedArray():
    content_float64 = awkward1.layout.NumpyArray(numpy.array([0.25, 0.5, 3.5, 4.5, 5.5], dtype=numpy.float64))
    array_float64 = awkward1.layout.UnmaskedArray(content_float64)
    assert awkward1.to_list(array_float64) == [0.25, 0.5, 3.5, 4.5, 5.5]
    assert str(awkward1.type(content_float64)) == "float64"
    assert str(awkward1.type(awkward1.Array(content_float64))) == "5 * float64"
    assert str(awkward1.type(array_float64)) == "?float64"
    assert str(awkward1.type(awkward1.Array(array_float64))) == "5 * ?float64"

    assert awkward1.can_cast(numpy.float32, numpy.float64) == True
    assert awkward1.can_cast(numpy.float64, numpy.float32, 'unsafe') == True
    assert awkward1.can_cast(numpy.float64, numpy.int8, 'unsafe') == True

    content_float32 = content_float64.numbers_to_type('float32')
    array_float32 = awkward1.layout.UnmaskedArray(content_float32)
    assert awkward1.to_list(array_float32) == [0.25, 0.5, 3.5, 4.5, 5.5]
    assert str(awkward1.type(content_float32)) == "float32"
    assert str(awkward1.type(awkward1.Array(content_float32))) == "5 * float32"
    assert str(awkward1.type(array_float32)) == "?float32"
    assert str(awkward1.type(awkward1.Array(array_float32))) == "5 * ?float32"

    content_int8 = content_float64.numbers_to_type('int8')
    array_int8 = awkward1.layout.UnmaskedArray(content_int8)
    assert awkward1.to_list(array_int8) == [0, 0, 3, 4, 5]
    assert str(awkward1.type(content_int8)) == "int8"
    assert str(awkward1.type(awkward1.Array(content_int8))) == "5 * int8"
    assert str(awkward1.type(array_int8)) == "?int8"
    assert str(awkward1.type(awkward1.Array(array_int8))) == "5 * ?int8"

    content_from_int8 = content_int8.numbers_to_type('float64')
    array_from_int8 = awkward1.layout.UnmaskedArray(content_from_int8)
    assert awkward1.to_list(array_from_int8) == [0, 0, 3, 4, 5]
    assert str(awkward1.type(content_from_int8)) == "float64"
    assert str(awkward1.type(awkward1.Array(content_from_int8))) == "5 * float64"
    assert str(awkward1.type(array_from_int8)) == "?float64"
    assert str(awkward1.type(awkward1.Array(array_from_int8))) == "5 * ?float64"

def test_RegularArray_and_ListArray():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]));
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
    regulararray = awkward1.layout.RegularArray(listoffsetarray, 2)
    starts = awkward1.layout.Index64(numpy.array([0, 1]))
    stops = awkward1.layout.Index64(numpy.array([2, 3]))
    listarray = awkward1.layout.ListArray64(starts, stops, regulararray)

    assert str(awkward1.type(content)) == "float64"
    assert str(awkward1.type(regulararray)) == "2 * var * float64"
    assert str(awkward1.type(listarray)) == "var * 2 * var * float64"

    regulararray_int8 = regulararray.numbers_to_type('int8')
    assert str(awkward1.type(regulararray_int8)) == "2 * var * int8"

    listarray_bool = listarray.numbers_to_type('bool')
    assert str(awkward1.type(listarray_bool)) == "var * 2 * var * bool"
