# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools

import pytest
import numpy

import awkward1
import awkward1.dressing.string

def test_fromnumpy():
    a = numpy.arange(2*3*5).reshape((2, 3, 5))
    b = awkward1.fromnumpy(a)
    assert awkward1.tolist(a) == awkward1.tolist(b)

def test_highlevel():
    a = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]])
    assert repr(a) == "<Array [[1.1 2.2 3.3] [] ... [7.7 8.8 9.9]] type='5 * var * float64'>"
    assert str(a) == "[[1.1 2.2 3.3] [] [4.4 5.5] [6.6] [7.7 8.8 9.9]]"

    b = awkward1.Array(numpy.arange(100, dtype=numpy.int32))
    assert repr(b) == "<Array [0 1 2 3 4 5 6 ... 93 94 95 96 97 98 99] type='100 * int32'>"
    assert str(b) == "[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 ... 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99]"

    c = awkward1.Array('[{"one": 3.14, "two": [1.1, 2.2]}, {"one": 99.9, "two": [-3.1415926]}]')
    assert repr(c) == "<Array [{one: 3.14, two: [1.1 ... -3.14]}] type='2 * {\"one\": float64, \"two\": var...'>"
    assert str(c) == "[{one: 3.14, two: [1.1 2.2]} {one: 99.9, two: [-3.14]}]"

class Dummy(awkward1.highlevel.Array):
    pass

def test_dress():
    dressed0 = awkward1.layout.DressedType(awkward1.layout.PrimitiveType("float64"), Dummy, {"one": 1, "two": 2})
    assert repr(dressed0) == "dress[float64, 'tests.test_PR028_add_dressed_types.Dummy', one=1, two=2]"

    pyclass = awkward1.dressing.string.String
    parameters = {"encoding": "utf-8"}
    inner = awkward1.layout.ListType(awkward1.layout.PrimitiveType("uint8"))
    dressed1 = awkward1.layout.DressedType(inner, pyclass, parameters)
    dressed2 = awkward1.layout.DressedType(inner, pyclass, parameters)
    dressed3 = awkward1.layout.DressedType(inner, pyclass)

    assert repr(dressed1) == "string"
    assert repr(dressed3) == "bytes"
    assert dressed1 == dressed2
    assert dressed1 != dressed3

def test_string():
    a = awkward1.Array(numpy.array([ord(x) for x in "hey there"], dtype=numpy.uint8))
    a.__class__ = awkward1.dressing.string.String
    assert str(a) == str(b"hey there")
    assert repr(a) == repr(b"hey there")
