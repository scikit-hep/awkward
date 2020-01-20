# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_basic():
    array = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    assert awkward1.tolist(array + array) == [[2.2, 4.4, 6.6], [], [8.8, 11.0]]
    assert awkward1.tolist(array * 2) == [[2.2, 4.4, 6.6], [], [8.8, 11.0]]

def test_emptyarray():
    one = awkward1.Array(awkward1.layout.NumpyArray(numpy.array([])))
    two = awkward1.Array(awkward1.layout.EmptyArray())
    assert awkward1.tolist(one + one) == []
    assert awkward1.tolist(two + two) == []
    assert awkward1.tolist(one + two) == []
