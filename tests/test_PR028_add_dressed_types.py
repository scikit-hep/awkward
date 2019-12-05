# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools

import pytest
import numpy

import awkward1
import awkward1.dressing.string

py27 = (sys.version_info[0] < 3)

def test_fromnumpy():
    a = numpy.arange(2*3*5).reshape((2, 3, 5))
    b = awkward1.fromnumpy(a)
    assert awkward1.tolist(a) == awkward1.tolist(b)

def test_highlevel():
    a = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]])
    assert repr(a) == "<Array [[1.1, 2.2, 3.3], ... [7.7, 8.8, 9.9]] type='5 * var * float64'>"
    assert str(a) == "[[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]"

    b = awkward1.Array(numpy.arange(100, dtype=numpy.int32))
    assert repr(b) == "<Array [0, 1, 2, 3, 4, ... 95, 96, 97, 98, 99] type='100 * int32'>"
    assert str(b) == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ... 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]"

    c = awkward1.Array('[{"one": 3.14, "two": [1.1, 2.2]}, {"one": 99.9, "two": [-3.1415926]}]')
    assert repr(c) == "<Array [{one: 3.14, two: [1.1, ... -3.14]}] type=\"2 * {'one': float64, 'two': va...\">"
    assert str(c) == "[{one: 3.14, two: [1.1, 2.2]}, {one: 99.9, two: [-3.14]}]"

class Dummy(awkward1.highlevel.Array):
    pass

def test_dress():
    dressed0 = awkward1.layout.DressedType(awkward1.layout.PrimitiveType("float64"), Dummy, one=1, two=2)
    assert repr(dressed0) in ("dress[float64, 'tests.test_PR028_add_dressed_types.Dummy', one=1, two=2]", "dress[float64, 'tests.test_PR028_add_dressed_types.Dummy', two=2, one=1]")

    pyclass = awkward1.dressing.string.String
    inner = awkward1.layout.PrimitiveType("uint8")

    baseline = sys.getrefcount(pyclass)
    assert (sys.getrefcount(pyclass), sys.getrefcount(inner)) == (baseline, 2)
    dressed1 = awkward1.layout.DressedType(inner, pyclass, encoding="utf-8")
    assert (sys.getrefcount(pyclass), sys.getrefcount(inner)) == (baseline + 1, 2)
    dressed2 = awkward1.layout.DressedType(inner, pyclass, encoding="utf-8")
    assert (sys.getrefcount(pyclass), sys.getrefcount(inner)) == (baseline + 2, 2)
    dressed3 = awkward1.layout.DressedType(inner, pyclass)
    assert (sys.getrefcount(pyclass), sys.getrefcount(inner)) == (baseline + 3, 2)

    assert repr(dressed1) == "string"
    assert repr(dressed3) == "bytes"
    assert dressed1 == dressed2
    assert dressed1 != dressed3

    assert (sys.getrefcount(pyclass), sys.getrefcount(inner)) == (baseline + 3, 2)
    del dressed1
    assert (sys.getrefcount(pyclass), sys.getrefcount(inner)) == (baseline + 2, 2)
    del dressed2
    assert (sys.getrefcount(pyclass), sys.getrefcount(inner)) == (baseline + 1, 2)
    del dressed3
    assert (sys.getrefcount(pyclass), sys.getrefcount(inner)) == (baseline, 2)

def test_string1():
    a = awkward1.Array(numpy.array([ord(x) for x in "hey there"], dtype=numpy.uint8))
    a.__class__ = awkward1.dressing.string.String
    assert str(a) == str(b"hey there")
    assert repr(a) == repr(b"hey there")

def test_string2():
    string = awkward1.layout.DressedType(awkward1.layout.PrimitiveType("uint8"), awkward1.dressing.string.String, encoding="utf-8")

    content = awkward1.layout.NumpyArray(numpy.array([ord(x) for x in "heythere"], dtype=numpy.uint8))
    content.type = string
    listoffsetarray = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 3, 3, 8])), content)
    a = awkward1.util.wrap(listoffsetarray)

    assert repr(a.layout.content.type) == "string"
    assert repr(a.layout.type) == "3 * string"

    if py27:
        assert repr(a) == "<Array [u'hey', u'', u'there'] type='3 * string'>"
    else:
        assert repr(a) == "<Array ['hey', '', 'there'] type='3 * string'>"
    assert repr(a[0]) == "u'hey'"
    assert repr(a[1]) == "u''"
    assert repr(a[2]) == "u'there'"
