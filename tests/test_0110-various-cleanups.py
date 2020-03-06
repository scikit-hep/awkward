# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_na_union():
    one = awkward1.Array([1, None, 3], checkvalid=True).layout
    two = awkward1.Array([[], [1], None, [3, 3, 3]], checkvalid=True).layout
    tags = awkward1.layout.Index8(numpy.array([0, 1, 1, 0, 0, 1, 1], dtype=numpy.int8))
    index = awkward1.layout.Index64(numpy.array([0, 0, 1, 1, 2, 2, 3], dtype=numpy.int64))
    array = awkward1.Array(awkward1.layout.UnionArray8_64(tags, index, [one, two]), checkvalid=True)
    assert awkward1.tolist(array) == [1, [], [1], None, 3, None, [3, 3, 3]]

    assert awkward1.tolist(awkward1.isna(array)) == [False, False, False, True, False, True, False]

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

    array = awkward1.Array(recordarray, behavior=behavior, checkvalid=True)
    assert repr(array) == "<DummyArray <1.1> <2.2> <3.3> <4.4> <5.5>>"
    assert repr(array[0]) == "<1.1>"

    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, recordarray)

    array2 = awkward1.Array(listoffsetarray, behavior=behavior, checkvalid=True)

    assert array2.layout.parameter("__record__") is None
    assert array2.layout.purelist_parameter("__record__") == "Dummy"

    assert repr(array2) == "<DeepDummyArray <DummyArray <1.1> <2.2> <3.3>> <DummyArray > <DummyArray <4.4> <5.5>>>"
    assert repr(array2[0]) == "<DummyArray <1.1> <2.2> <3.3>>"
    assert repr(array2[0, 0]) == "<1.1>"

    recordarray2 = awkward1.layout.RecordArray({"outer": listoffsetarray})

    array3 = awkward1.Array(recordarray2, behavior=behavior, checkvalid=True)
    assert type(array3) is awkward1.Array
    assert type(array3["outer"]) is DeepDummyArray
    assert repr(array3["outer"]) == "<DeepDummyArray <DummyArray <1.1> <2.2> <3.3>> <DummyArray > <DummyArray <4.4> <5.5>>>"

def test_sizes():
    # FIXME: this should return a scalar 6, not NumpyArray([6])
    # print(awkward1.sizes(awkward1.Array(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]), checkvalid=True), axis=0))
    assert awkward1.tolist(awkward1.sizes(awkward1.Array(numpy.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]), checkvalid=True), axis=0)) == [3, 3]
    assert awkward1.tolist(awkward1.sizes(awkward1.Array(numpy.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]]), checkvalid=True), axis=0)) == [2, 2]
    assert awkward1.tolist(awkward1.sizes(awkward1.Array(numpy.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]]), checkvalid=True), axis=1)) == [[2, 2], [2, 2]]
    # FIXME: this should not be possible, as opposed to returning NumpyArray([])
    # print(awkward1.sizes(awkward1.Array(numpy.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]]), checkvalid=True), axis=2))
    assert awkward1.tolist(awkward1.sizes(awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], checkvalid=True), axis=0)) == [3, 0, 2]

def test_flatten():
    assert awkward1.tolist(awkward1.flatten(awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], checkvalid=True), axis=0)) == [1.1, 2.2, 3.3, 4.4, 5.5]

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
        assert awkward1.tolist(awkward1.Array(left, checkvalid=True) == awkward1.Array(right, checkvalid=True)) == [x == y for x, y in zip(left, right)]

def test_string_equal2():
    assert awkward1.tolist(awkward1.Array(["one", "two", "three", "two", "two", "one"], checkvalid=True) == "two") == [False, True, False, True, True, False]
