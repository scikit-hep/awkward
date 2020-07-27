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
