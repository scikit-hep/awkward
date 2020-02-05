# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_na_union():
    one = awkward1.Array([1, None, 3]).layout
    two = awkward1.Array([[], [1], None, [3, 3, 3]]).layout
    tags = awkward1.layout.Index8(numpy.array([0, 1, 1, 0, 0, 1, 1], dtype=numpy.int8))
    index = awkward1.layout.Index64(numpy.array([0, 0, 1, 1, 2, 2, 3], dtype=numpy.int64))
    array = awkward1.Array(awkward1.layout.UnionArray8_64(tags, index, [one, two]))
    assert awkward1.tolist(array) == [1, [], [1], None, 3, None, [3, 3, 3]]

    assert awkward1.tolist(awkward1.isna(array)) == [False, False, False, True, False, True, False]

class DummyRecord(awkward1.Record):
    def __repr__(self):
        return "<{0}>".format(self.x)

class DummyArray(awkward1.Array):
    def __repr__(self):
        return "<DummyArray {0}>".format(" ".join(repr(x) for x in self))

def test_behaviors():
    behavior = {}
    behavior["Dummy"] = DummyRecord
    behavior["array", "Dummy"] = DummyArray

    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    recordarray = awkward1.layout.RecordArray({"x": content})
    recordarray.setparameter("__record__", "Dummy")

    array = awkward1.Array(recordarray, behavior=behavior)
    assert repr(array) == "<DummyArray <1.1> <2.2> <3.3> <4.4> <5.5>>"
    assert repr(array[0]) == "<1.1>"

    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, recordarray)

    array2 = awkward1.Array(listoffsetarray, behavior=behavior)
    print(repr(array2))
    assert repr(array2[0]) == "<DummyArray <1.1> <2.2> <3.3>>"
    assert repr(array2[0, 0]) == "<1.1>"

    # raise Exception
