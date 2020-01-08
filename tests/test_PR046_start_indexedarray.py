# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_basic():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4]))

    ind = numpy.array([2, 2, 0, 3, 4], dtype=numpy.int32)
    index = awkward1.layout.Index32(ind)
    array = awkward1.layout.IndexedArray32(index, content)
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = numpy.array([2, 2, 0, 3, 4], dtype=numpy.uint32)
    index = awkward1.layout.IndexU32(ind)
    array = awkward1.layout.IndexedArrayU32(index, content)
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = numpy.array([2, 2, 0, 3, 4], dtype=numpy.int64)
    index = awkward1.layout.Index64(ind)
    array = awkward1.layout.IndexedArray64(index, content)
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = numpy.array([2, 2, 0, 3, 4], dtype=numpy.int32)
    index = awkward1.layout.Index32(ind)
    array = awkward1.layout.IndexedOptionArray32(index, content)
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = numpy.array([2, 2, 0, 3, 4], dtype=numpy.int64)
    index = awkward1.layout.Index64(ind)
    array = awkward1.layout.IndexedOptionArray64(index, content)
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

def test_null():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4]))
    index = awkward1.layout.Index64(numpy.array([2, 2, 0, -1, 4], dtype=numpy.int64))
    array = awkward1.layout.IndexedOptionArray64(index, content)
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, None, 4.4]

def test_carry():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4]))
    index = awkward1.layout.Index64(numpy.array([2, 2, 0, 3, 4], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray64(index, content)
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, indexedarray)
    assert awkward1.tolist(listoffsetarray) == [[2.2, 2.2, 0.0], [], [3.3, 4.4]]
    assert awkward1.tolist(listoffsetarray[::-1]) == [[3.3, 4.4], [], [2.2, 2.2, 0.0]]
    assert awkward1.tolist(listoffsetarray[[2, 0]]) == [[3.3, 4.4], [2.2, 2.2, 0.0]]
    assert awkward1.tolist(listoffsetarray[[2, 0], 1]) == [4.4, 2.2]     # invokes carry
    assert awkward1.tolist(listoffsetarray[2:, 1]) == [4.4]              # invokes carry

    index = awkward1.layout.Index64(numpy.array([2, 2, 0, 3, -1], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedOptionArray64(index, content)
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, indexedarray)
    assert awkward1.tolist(listoffsetarray) == [[2.2, 2.2, 0.0], [], [3.3, None]]
    assert awkward1.tolist(listoffsetarray[::-1]) == [[3.3, None], [], [2.2, 2.2, 0.0]]
    assert awkward1.tolist(listoffsetarray[[2, 0]]) == [[3.3, None], [2.2, 2.2, 0.0]]
    assert awkward1.tolist(listoffsetarray[[2, 0], 1]) == [None, 2.2]    # invokes carry
    assert awkward1.tolist(listoffsetarray[2:, 1]) == [None]             # invokes carry

def test_others():
    content = awkward1.layout.NumpyArray(numpy.array([[0.0, 0.0], [0.1, 1.0], [0.2, 2.0], [0.3, 3.0], [0.4, 4.0], [0.5, 5.0]]))
    index = awkward1.layout.Index64(numpy.array([4, 0, 3, 1, 3], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray64(index, content)
    assert indexedarray[3, 0] == 0.1
    assert indexedarray[3, 1] == 1.0
    assert awkward1.tolist(indexedarray[3, ::-1]) == [1.0, 0.1]
    assert awkward1.tolist(indexedarray[3, [1, 1, 0]]) == [1.0, 1.0, 0.1]
