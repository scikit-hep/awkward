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
