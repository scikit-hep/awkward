# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_sort_emptyarray():
    array = awkward1.layout.EmptyArray()
    assert awkward1.to_list(array.sort(0, True, False)) == []
    assert awkward1.to_list(array.argsort(0, True, False)) == []

def test_sort_numpyarray():
    array = awkward1.layout.NumpyArray(numpy.array([3, 2, 1, 5, 4]))
    assert awkward1.to_list(array.argsort(0, True, False)) == [2, 1, 0, 4, 3]
    assert awkward1.to_list(array.argsort(0, False, False)) == [3, 4, 0, 1, 2]

    assert awkward1.to_list(array.sort(0, True, False)) == [1, 2, 3, 4, 5]
    assert awkward1.to_list(array.sort(0, False, False)) == [5, 4, 3, 2, 1]

    array = awkward1.layout.NumpyArray(numpy.array([[3.3, 2.2, 4.4], [1.1, 5.5, 3.3]]))
    assert awkward1.to_list(array.sort(0, True, False)) == [[2.2, 3.3, 4.4], [1.1, 3.3, 5.5]]
    assert awkward1.to_list(array.sort(1, True, False)) == [[1.1, 2.2, 3.3], [3.3, 4.4, 5.5]]

def test_sort_indexoffsetarray():
    array = awkward1.Array([[2.2, 1.1, 3.3], [], [4.4, 5.5], [5.5], [-4.4, -5.5, -6.6]]).layout
    assert awkward1.to_list(array.argsort(0, True, False)) == [[1, 0, 2], [None], [0, 1], [0], [2, 1, 0]]
    assert awkward1.to_list(array.argsort(1, True, False)) == [9, 8, 7, 1, 0, 2, 4, 5, 6, 3]
