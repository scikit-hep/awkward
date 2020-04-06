# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_sort_emptyarray():
    array = awkward1.layout.EmptyArray()
    assert awkward1.to_list(array.sort(0, True, False)) == []

def test_sort_numpyarray():
    array = awkward1.layout.NumpyArray(numpy.array([3, 2, 1, 5, 4]))
    assert awkward1.to_list(array.argsort(0, True, False)) == [2, 1, 0, 4, 3]
    assert awkward1.to_list(array.argsort(0, False, False)) == [3, 4, 0, 1, 2]

    assert awkward1.to_list(array.sort(0, True, False)) == [1, 2, 3, 4, 5]
    assert awkward1.to_list(array.sort(0, False, False)) == [5, 4, 3, 2, 1]

    # np.sort([[3.3, 2.2, 4.4], [1.1, 5.5, 3.3]], axis=-1)
    # array([[2.2, 3.3, 4.4],
    #        [1.1, 3.3, 5.5]])

    # np.sort([[3.3, 2.2, 4.4], [1.1, 5.5, 3.3]], axis=-2)
    # array([[1.1, 2.2, 3.3],
    #        [3.3, 5.5, 4.4]])
    array = awkward1.layout.NumpyArray(numpy.array([[2.2, 1.1, 3.3], [6.6, 4.4, 5.5]]))
    assert awkward1.to_list(array.sort(0, True, False)) == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]
    assert awkward1.to_list(array.sort(1, True, False)) == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

def test_sort_indexoffsetarray():
    array = awkward1.Array([[2.2, 1.1, 3.3], [], [4.4, 5.5], [5.5], [-4.4, -5.5, -6.6]]).layout
    assert awkward1.to_list(array.argsort(1, True, False)) == [1, None, 0, 0, 2]

    index2 = awkward1.layout.Index64(numpy.array([4, 3, 2, 1, 0], dtype=numpy.int64))
    array2 = awkward1.layout.IndexedArray64(index2, array)
    assert awkward1.to_list(array2.argsort(1, True, False)) == [2, 0, 0, None, 1]

    index3 = awkward1.layout.Index64(numpy.array([4, 3, -1, 4, 0], dtype=numpy.int64))
    array2 = awkward1.layout.IndexedArray64(index3, array)
    assert awkward1.to_list(array2.argsort(1, True, False)) == [2, 0, None, 2, 1]
