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

    assert awkward1.tolist(array[awkward1.layout.EmptyArray()]) == []

    content0 = awkward1.layout.NumpyArray(numpy.array([5, 2, 2]))
    content1 = awkward1.layout.NumpyArray(numpy.array([3, 9, 0, 1]))
    tags = awkward1.layout.Index8(numpy.array([0, 0, 0, 1, 1, 1, 1], dtype=numpy.int8))
    index2 = awkward1.layout.Index64(numpy.array([0, 1, 2, 0, 1, 2, 3], dtype=numpy.int64))
    unionarray = awkward1.layout.UnionArray8_64(tags, index2, [content0, content1])
    assert awkward1.tolist(array[unionarray]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert awkward1.tolist(array[awkward1.Array(unionarray)]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]

    array = awkward1.Array(numpy.array([[0.0, 1.1, 2.2, 3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]))
    assert awkward1.tolist(array[awkward1.layout.NumpyArray(numpy.array([[0, 1], [1, 0]])), awkward1.layout.NumpyArray(numpy.array([[2, 4], [3, 3]]))]) == [[2.2, 9.9], [8.8, 3.3]]
    assert awkward1.tolist(array[awkward1.layout.NumpyArray(numpy.array([[0, 1], [1, 0]]))]) == [[[0.0, 1.1, 2.2, 3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]], [[5.5, 6.6, 7.7, 8.8, 9.9], [0.0, 1.1, 2.2, 3.3, 4.4]]]

    array = awkward1.Array([{"x": 1, "y": 1.1, "z": [1]}, {"x": 2, "y": 2.2, "z": [2, 2]}, {"x": 3, "y": 3.3, "z": [3, 3, 3]}, {"x": 4, "y": 4.4, "z": [4, 4, 4, 4]}, {"x": 5, "y": 5.5, "z": [5, 5, 5, 5, 5]}])
    awkward1.tolist(array[awkward1.Array(["y", "x"]).layout]) == [{"y": 1.1, "x": 1}, {"y": 2.2, "x": 2}, {"y": 3.3, "x": 3}, {"y": 4.4, "x": 4}, {"y": 5.5, "x": 5}]

# def test_new_slices():
#     content = awkward1.layout.NumpyArray(numpy.array([1, 0, 9, 3, 2, 2, 5], dtype=numpy.int64))
#     index = awkward1.layout.Index64(numpy.array([6, 5, -1, 3, 2, -1, 0], dtype=numpy.int64))
#     indexedarray = awkward1.layout.IndexedOptionArray64(index, content)
#     assert awkward1.tolist(indexedarray) == [5, 2, None, 3, 9, None, 1]

#     assert repr(awkward1.layout.Slice(indexedarray)) == "[missing([0, 1, -1, ..., 3, -1, 4], array([5, 2, 3, 9, 1]))]"

#     offsets = awkward1.layout.Index64(numpy.array([0, 4, 4, 7], dtype=numpy.int64))
#     listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
#     assert awkward1.tolist(listoffsetarray) == [[1, 0, 9, 3], [], [2, 2, 5]]

#     repr(awkward1.layout.Slice(listoffsetarray)) == "[jagged([0, 3, 3, 5], array([0, 9, 3, 2, 2]))]"

#     offsets = awkward1.layout.Index64(numpy.array([1, 4, 4, 6], dtype=numpy.int64))
#     listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
#     assert awkward1.tolist(listoffsetarray) == [[0, 9, 3], [], [2, 2]]

#     repr(awkward1.layout.Slice(listoffsetarray)) == "[jagged([0, 3, 3, 5], array([0, 9, 3, 2, 2]))]"

#     starts = awkward1.layout.Index64(numpy.array([1, 99, 5], dtype=numpy.int64))
#     stops = awkward1.layout.Index64(numpy.array([4, 99, 7], dtype=numpy.int64))
#     listarray = awkward1.layout.ListArray64(starts, stops, content)
#     assert awkward1.tolist(listarray) == [[0, 9, 3], [], [2, 5]]

#     repr(awkward1.layout.Slice(listoffsetarray)) == "[jagged([0, 3, 3, 5], array([0, 9, 3, 2, 5]))]"

def test_missing():
    array = awkward1.Array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])

    # print(array[awkward1.Array([3, 6, None, None, 8, 6])])
    # raise Exception
