# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import itertools
import collections

import pytest
import numpy

import awkward1
import awkward1.behaviors.string

py27 = (sys.version_info[0] < 3)

def test_fromnumpy():
    a = numpy.arange(2*3*5).reshape((2, 3, 5))
    b = awkward1.from_numpy(a)
    assert awkward1.to_list(a) == awkward1.to_list(b)

def test_highlevel():
    a = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], check_valid=True)
    assert repr(a) == "<Array [[1.1, 2.2, 3.3], ... [7.7, 8.8, 9.9]] type='5 * var * float64'>"
    assert str(a) == "[[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]"

    b = awkward1.Array(numpy.arange(100, dtype=numpy.int32), check_valid=True)
    assert repr(b) == "<Array [0, 1, 2, 3, 4, ... 95, 96, 97, 98, 99] type='100 * int32'>"
    assert str(b) == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ... 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]"

    c = awkward1.Array('[{"one": 3.14, "two": [1.1, 2.2]}, {"one": 99.9, "two": [-3.1415926]}]', check_valid=True)
    assert repr(c) == "<Array [{one: 3.14, two: [1.1, ... -3.14]}] type='2 * {\"one\": float64, \"two\": va...'>"
    assert str(c) == "[{one: 3.14, two: [1.1, 2.2]}, {one: 99.9, two: [-3.14]}]"

class Dummy(awkward1.highlevel.Array):
    pass

def test_string1():
    a = awkward1.Array(numpy.array([ord(x) for x in "hey there"], dtype=numpy.uint8), check_valid=True)
    a.__class__ = awkward1.behaviors.string.ByteBehavior
    assert str(a) == str(b"hey there")
    assert repr(a) == repr(b"hey there")

def test_string2():
    content = awkward1.layout.NumpyArray(numpy.array([ord(x) for x in "heythere"], dtype=numpy.uint8))
    listoffsetarray = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 3, 3, 8])), content)
    a = awkward1.Array(listoffsetarray, check_valid=True)

    assert isinstance(a, awkward1.Array)
    assert not isinstance(a, awkward1.behaviors.string.StringBehavior)
    assert awkward1.to_list(a) == [[104, 101, 121], [], [116, 104, 101, 114, 101]]

    assert repr(awkward1.type(a)) == "3 * var * uint8"
    assert repr(awkward1.type(a[0])) == "3 * uint8"
    assert repr(awkward1.type(a[1])) == "0 * uint8"
    assert repr(awkward1.type(a[2])) == "5 * uint8"

    assert repr(a) == "<Array [[104, 101, 121], ... 101, 114, 101]] type='3 * var * uint8'>"
    assert str(a) == "[[104, 101, 121], [], [116, 104, 101, 114, 101]]"
    assert repr(a[0]) == "<Array [104, 101, 121] type='3 * uint8'>"
    assert repr(a[1]) == "<Array [] type='0 * uint8'>"
    assert repr(a[2]) == "<Array [116, 104, 101, 114, 101] type='5 * uint8'>"

    content = awkward1.layout.NumpyArray(numpy.array([ord(x) for x in "heythere"], dtype=numpy.uint8), parameters={"__array__": "char", "encoding": "utf-8"})
    listoffsetarray = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 3, 3, 8])), content, parameters={"__array__": "string"})
    a = awkward1.Array(listoffsetarray, check_valid=True)

    a = awkward1.Array(listoffsetarray, check_valid=True)
    assert isinstance(a, awkward1.Array)
    assert awkward1.to_list(a) == ['hey', '', 'there']

    if py27:
        assert str(a) == "[u'hey', u'', u'there']"
        assert repr(a[0]) == "u'hey'"
        assert repr(a[1]) == "u''"
        assert repr(a[2]) == "u'there'"
    else:
        assert str(a) == "['hey', '', 'there']"
        assert repr(a[0]) == "'hey'"
        assert repr(a[1]) == "''"
        assert repr(a[2]) == "'there'"
