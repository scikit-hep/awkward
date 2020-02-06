# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_array_slice():
    array = awkward1.Array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    assert awkward1.tolist(array[[5, 2, 2, 3, 9, 0, 1]]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert awkward1.tolist(array[numpy.array([5, 2, 2, 3, 9, 0, 1])]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert awkward1.tolist(array[awkward1.layout.NumpyArray(numpy.array([5, 2, 2, 3, 9, 0, 1], dtype=numpy.int32))]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert awkward1.tolist(array[awkward1.Array(numpy.array([5, 2, 2, 3, 9, 0, 1], dtype=numpy.int32))]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert awkward1.tolist(array[awkward1.Array([5, 2, 2, 3, 9, 0, 1])]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]

    assert awkward1.tolist(array[awkward1.layout.NumpyArray(numpy.array([False, False, False, False, False, True, False, True, False, True]))]) == [5.5, 7.7, 9.9]

    content = awkward1.layout.NumpyArray(numpy.array([1, 0, 9, 3, 2, 2, 5], dtype=numpy.int64))
    index = awkward1.layout.Index64(numpy.array([6, 5, 4, 3, 2, 1, 0], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray64(index, content)
    assert awkward1.tolist(array[indexedarray]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert awkward1.tolist(array[awkward1.Array(indexedarray)]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]

# TEST: awkward array of strings
