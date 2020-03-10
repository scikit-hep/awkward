# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_fillna_empty_array():
    empty = awkward1.layout.EmptyArray()

    assert awkward1.tolist(empty) == []
    array = empty.rpad(5, 0)
    assert awkward1.tolist(array) == [None, None, None, None, None]
    assert awkward1.tolist(array.fillna(10)) == [10, 10, 10, 10, 10]

def test_fillna_numpy_array():
    content = awkward1.layout.NumpyArray(numpy.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]))
    array = content.rpad(3,0)
    assert awkward1.tolist(array) == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], None]
    assert awkward1.tolist(array.fillna(0)) == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], 0]

# def test_fillna_numpy_O_array():
#     optarray = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, None, 5.5, 6.6, None]))
#     assert awkward1.tolist(optarray.fillna(0)) == [1.1, 2.2, 3.3, 0, 5.5, 6.6, 0]

def test_fillna_regular_array():
    content = awkward1.layout.NumpyArray(numpy.array([2.1, 8.4, 7.4, 1.6, 2.2, 3.4, 6.2, 5.4, 1.5, 3.9, 3.8, 3.0, 8.5, 6.9, 4.3, 3.6, 6.7, 1.8, 3.2]))
    index = awkward1.layout.Index64(numpy.array([13, 9, 13, 4, 8, 3, 15, -1, 16, 2, 8], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedOptionArray64(index, content)
    regarray = awkward1.layout.RegularArray(indexedarray, 3)

    assert awkward1.tolist(regarray) == [[6.9, 3.9, 6.9], [2.2, 1.5, 1.6], [3.6, None, 6.7]]
    assert awkward1.tolist(regarray.fillna(666)) ==  [[6.9, 3.9, 6.9], [2.2, 1.5, 1.6], [3.6, 666, 6.7]]

def test_fillna_listarray_array():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, None, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    starts  = awkward1.layout.Index64(numpy.array([0, 3, 4, 5, 8]))
    stops   = awkward1.layout.Index64(numpy.array([3, 3, 6, 8, 9]))
    listarray   = awkward1.layout.ListArray64(starts, stops, content)

    assert awkward1.tolist(listarray) == [[0.0, 1.1, None], [], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8]]
    assert awkward1.tolist(listarray.fillna(-1000)) == [[0.0, 1.1, -1000], [], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8]]
