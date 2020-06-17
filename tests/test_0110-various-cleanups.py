# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_na_union():
    one = awkward1.from_iter([1, None, 3], highlevel=False)
    two = awkward1.from_iter([[], [1], None, [3, 3, 3]], highlevel=False)
    tags = awkward1.layout.Index8(numpy.array([0, 1, 1, 0, 0, 1, 1], dtype=numpy.int8))
    index = awkward1.layout.Index64(numpy.array([0, 0, 1, 1, 2, 2, 3], dtype=numpy.int64))
    array = awkward1.Array(awkward1.layout.UnionArray8_64(tags, index, [one, two]), check_valid=True)
    assert awkward1.to_list(array) == [1, [], [1], None, 3, None, [3, 3, 3]]

    assert awkward1.to_list(awkward1.is_none(array)) == [False, False, False, True, False, True, False]

class DummyRecord(awkward1.Record):
    def __repr__(self):
        return "<{0}>".format(self.x)

class DummyArray(awkward1.Array):
    def __repr__(self):
        return "<DummyArray {0}>".format(" ".join(repr(x) for x in self))

class DeepDummyArray(awkward1.Array):
    def __repr__(self):
        return "<DeepDummyArray {0}>".format(" ".join(repr(x) for x in self))

def test_behaviors():
    behavior = {}
    behavior["Dummy"] = DummyRecord
    behavior[".", "Dummy"] = DummyArray
    behavior["*", "Dummy"] = DeepDummyArray

    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    recordarray = awkward1.layout.RecordArray({"x": content})
    recordarray.setparameter("__record__", "Dummy")

    array = awkward1.Array(recordarray, behavior=behavior, check_valid=True)
    assert repr(array) == "<DummyArray <1.1> <2.2> <3.3> <4.4> <5.5>>"
    assert repr(array[0]) == "<1.1>"

    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, recordarray)

    array2 = awkward1.Array(listoffsetarray, behavior=behavior, check_valid=True)

    assert array2.layout.parameter("__record__") is None
    assert array2.layout.purelist_parameter("__record__") == "Dummy"

    assert repr(array2) == "<DeepDummyArray <DummyArray <1.1> <2.2> <3.3>> <DummyArray > <DummyArray <4.4> <5.5>>>"
    assert repr(array2[0]) == "<DummyArray <1.1> <2.2> <3.3>>"
    assert repr(array2[0, 0]) == "<1.1>"

    recordarray2 = awkward1.layout.RecordArray({"outer": listoffsetarray})

    array3 = awkward1.Array(recordarray2, behavior=behavior, check_valid=True)
    assert type(array3) is awkward1.Array
    assert type(array3["outer"]) is DeepDummyArray
    assert repr(array3["outer"]) == "<DeepDummyArray <DummyArray <1.1> <2.2> <3.3>> <DummyArray > <DummyArray <4.4> <5.5>>>"

def test_flatten():
    assert awkward1.to_list(awkward1.flatten(awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True), axis=1)) == [1.1, 2.2, 3.3, 4.4, 5.5]

def test_string_equal():
    trials = [
        (["one", "two", "", "three", "four", "", "five", "six", ""],
         ["one", "two", "", "three", "four", "", "five", "six", ""]),

        (["one", "two", "", "three", "four", "", "five", "six"],
         ["one", "two", "", "three", "four", "", "five", "six"]),

        (["one", "two", "", "three", "four", "", "five", "six", ""],
         ["one", "Two", "", "threE", "four", "", "five", "siX", ""]),

        (["one", "two", "", "three", "four", "", "five", "six"],
         ["one", "Two", "", "threE", "four", "", "five", "siX"]),

        (["one", "two", "", "thre", "four", "", "five", "six", ""],
         ["one", "two", "", "three", "four", "", "five", "six", ""]),

        (["one", "two", "", "thre", "four", "", "five", "six"],
         ["one", "two", "", "three", "four", "", "five", "six"]),

        (["one", "two", "", "three", "four", "", "five", "six", ""],
         ["one", "two", ":)", "three", "four", "", "five", "six", ""]),

        (["one", "two", "", "three", "four", "", "five", "six"],
         ["one", "two", ":)", "three", "four", "", "five", "six"]),

        (["one", "two", "", "three", "four", "", "five", "six", ""],
         ["", "two", "", "three", "four", "", "five", "six", ""]),

        (["one", "two", "", "three", "four", "", "five", "six"],
         ["", "two", "", "three", "four", "", "five", "six"]),
        ]

    for left, right in trials:
        assert awkward1.to_list(awkward1.Array(left, check_valid=True) == awkward1.Array(right, check_valid=True)) == [x == y for x, y in zip(left, right)]

def test_string_equal2():
    assert awkward1.to_list(awkward1.Array(["one", "two", "three", "two", "two", "one"], check_valid=True) == "two") == [False, True, False, True, True, False]
