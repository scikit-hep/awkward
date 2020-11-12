# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_basic():
    array = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)
    assert awkward1.to_list(array + array) == [[2.2, 4.4, 6.6], [], [8.8, 11.0]]
    assert awkward1.to_list(array * 2) == [[2.2, 4.4, 6.6], [], [8.8, 11.0]]

def test_emptyarray():
    one = awkward1.Array(awkward1.layout.NumpyArray(numpy.array([])), check_valid=True)
    two = awkward1.Array(awkward1.layout.EmptyArray(), check_valid=True)
    assert awkward1.to_list(one + one) == []
    assert awkward1.to_list(two + two) == []
    assert awkward1.to_list(one + two) == []

def test_indexedarray():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index1 = awkward1.layout.Index64(numpy.array([2, 4, 4, 0, 8], dtype=numpy.int64))
    index2 = awkward1.layout.Index64(numpy.array([6, 4, 4, 8, 0], dtype=numpy.int64))
    one = awkward1.Array(awkward1.layout.IndexedArray64(index1, content), check_valid=True)
    two = awkward1.Array(awkward1.layout.IndexedArray64(index2, content), check_valid=True)
    assert awkward1.to_list(one + two) == [8.8, 8.8, 8.8, 8.8, 8.8]

def test_indexedoptionarray():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index1 = awkward1.layout.Index64(numpy.array([2, -1, 4, 0, 8], dtype=numpy.int64))
    index2 = awkward1.layout.Index64(numpy.array([-1, 4, 4, -1, 0], dtype=numpy.int64))
    one = awkward1.Array(awkward1.layout.IndexedOptionArray64(index1, content), check_valid=True)
    two = awkward1.Array(awkward1.layout.IndexedOptionArray64(index2, content), check_valid=True)
    assert awkward1.to_list(one + two) == [None, None, 8.8, None, 8.8]

    uno = awkward1.layout.NumpyArray(numpy.array([2.2, 4.4, 4.4, 0.0, 8.8]))
    dos = awkward1.layout.NumpyArray(numpy.array([6.6, 4.4, 4.4, 8.8, 0.0]))
    assert awkward1.to_list(uno + two) == [None, 8.8, 8.8, None, 8.8]
    assert awkward1.to_list(one + dos) == [8.8, None, 8.8, 8.8, 8.8]

def test_regularize_shape():
    array = awkward1.layout.NumpyArray(numpy.arange(2*3*5).reshape(2, 3, 5))
    assert isinstance(array.toRegularArray(), awkward1.layout.RegularArray)
    assert awkward1.to_list(array.toRegularArray()) == awkward1.to_list(array)

def test_regulararray():
    array = awkward1.Array(numpy.arange(2*3*5).reshape(2, 3, 5), check_valid=True)
    assert awkward1.to_list(array + array) == (numpy.arange(2*3*5).reshape(2, 3, 5) * 2).tolist()
    assert awkward1.to_list(array * 2) == (numpy.arange(2*3*5).reshape(2, 3, 5) * 2).tolist()
    array2 = awkward1.Array(numpy.arange(2*1*5).reshape(2, 1, 5), check_valid=True)
    assert awkward1.to_list(array + array2) == awkward1.to_list(numpy.arange(2*3*5).reshape(2, 3, 5) + numpy.arange(2*1*5).reshape(2, 1, 5))
    array3 = awkward1.Array(numpy.arange(2*3*5).reshape(2, 3, 5).tolist(), check_valid=True)
    assert awkward1.to_list(array + array3) == awkward1.to_list(numpy.arange(2*3*5).reshape(2, 3, 5) + numpy.arange(2*3*5).reshape(2, 3, 5))
    assert awkward1.to_list(array3 + array) == awkward1.to_list(numpy.arange(2*3*5).reshape(2, 3, 5) + numpy.arange(2*3*5).reshape(2, 3, 5))

def test_listarray():
    content = awkward1.layout.NumpyArray(numpy.arange(12, dtype=numpy.int64))
    starts = awkward1.layout.Index64(numpy.array([3, 0, 999, 2, 6, 10], dtype=numpy.int64))
    stops  = awkward1.layout.Index64(numpy.array([7, 3, 999, 4, 6, 12], dtype=numpy.int64))
    one = awkward1.Array(awkward1.layout.ListArray64(starts, stops, content), check_valid=True)
    two = awkward1.Array([[100, 100, 100, 100], [200, 200, 200], [], [300, 300], [], [400, 400]], check_valid=True)
    assert awkward1.to_list(one) == [[3, 4, 5, 6], [0, 1, 2], [], [2, 3], [], [10, 11]]
    assert awkward1.to_list(one + 100) == [[103, 104, 105, 106], [100, 101, 102], [], [102, 103], [], [110, 111]]
    assert awkward1.to_list(one + two) == [[103, 104, 105, 106], [200, 201, 202], [], [302, 303], [], [410, 411]]
    assert awkward1.to_list(two + one) == [[103, 104, 105, 106], [200, 201, 202], [], [302, 303], [], [410, 411]]
    assert awkward1.to_list(one + numpy.array([100, 200, 300, 400, 500, 600])[:, numpy.newaxis]) == [[103, 104, 105, 106], [200, 201, 202], [], [402, 403], [], [610, 611]]
    assert awkward1.to_list(numpy.array([100, 200, 300, 400, 500, 600])[:, numpy.newaxis] + one) == [[103, 104, 105, 106], [200, 201, 202], [], [402, 403], [], [610, 611]]
    assert awkward1.to_list(one + 100) == [[103, 104, 105, 106], [100, 101, 102], [], [102, 103], [], [110, 111]]

def test_unionarray():
    one0 = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3], dtype=numpy.float64))
    one1 = awkward1.layout.NumpyArray(numpy.array([4, 5], dtype=numpy.int64))
    onetags = awkward1.layout.Index8(numpy.array([0, 0, 0, 0, 1, 1], dtype=numpy.int8))
    oneindex = awkward1.layout.Index64(numpy.array([0, 1, 2, 3, 0, 1], dtype=numpy.int64))
    one = awkward1.Array(awkward1.layout.UnionArray8_64(onetags, oneindex, [one0, one1]), check_valid=True)

    two0 = awkward1.layout.NumpyArray(numpy.array([0, 100], dtype=numpy.int64))
    two1 = awkward1.layout.NumpyArray(numpy.array([200.3, 300.3, 400.4, 500.5], dtype=numpy.float64))
    twotags = awkward1.layout.Index8(numpy.array([0, 0, 1, 1, 1, 1], dtype=numpy.int8))
    twoindex = awkward1.layout.Index64(numpy.array([0, 1, 0, 1, 2, 3], dtype=numpy.int64))
    two = awkward1.Array(awkward1.layout.UnionArray8_64(twotags, twoindex, [two0, two1]), check_valid=True)

    assert awkward1.to_list(one) == [0.0, 1.1, 2.2, 3.3, 4, 5]
    assert awkward1.to_list(two) == [0, 100, 200.3, 300.3, 400.4, 500.5]
    assert awkward1.to_list(one + two) == [0.0, 101.1, 202.5, 303.6, 404.4, 505.5]
