# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_fillna_empty_array():
    empty = awkward1.layout.EmptyArray()

    assert awkward1.to_list(empty) == []
    array = empty.rpad(5, 0)
    assert awkward1.to_list(array) == [None, None, None, None, None]
    value = awkward1.layout.NumpyArray(numpy.array([10]))
    assert awkward1.to_list(array.fillna(value)) == [10, 10, 10, 10, 10]

def test_fillna_numpy_array():
    content = awkward1.layout.NumpyArray(numpy.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]))
    array = content.rpad(3,0)
    assert awkward1.to_list(array) == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], None]
    value = awkward1.layout.NumpyArray(numpy.array([0]))
    assert awkward1.to_list(array.fillna(value)) == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], 0]

    array = content.rpad(5,1)
    assert awkward1.to_list(array) == [[1.1, 2.2, 3.3, None, None], [4.4, 5.5, 6.6, None, None]]
    value = awkward1.layout.NumpyArray(numpy.array([0]))
    assert awkward1.to_list(array.fillna(value)) == [[1.1, 2.2, 3.3, 0, 0], [4.4, 5.5, 6.6, 0, 0]]

# def test_fillna_numpy_O_array():
#     pyobject_array = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, None, 5.5, 6.6, None]))
#     assert awkward1.to_list(optarray.fillna(0)) == [1.1, 2.2, 3.3, 0, 5.5, 6.6, 0]

def test_fillna_regular_array():
    content = awkward1.layout.NumpyArray(numpy.array([2.1, 8.4, 7.4, 1.6, 2.2, 3.4, 6.2, 5.4, 1.5, 3.9, 3.8, 3.0, 8.5, 6.9, 4.3, 3.6, 6.7, 1.8, 3.2]))
    index = awkward1.layout.Index64(numpy.array([13, 9, 13, 4, 8, 3, 15, -1, 16, 2, 8], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedOptionArray64(index, content)
    regarray = awkward1.layout.RegularArray(indexedarray, 3)

    assert awkward1.to_list(regarray) == [[6.9, 3.9, 6.9], [2.2, 1.5, 1.6], [3.6, None, 6.7]]
    value = awkward1.layout.NumpyArray(numpy.array([666]))
    assert awkward1.to_list(regarray.fillna(value)) ==  [[6.9, 3.9, 6.9], [2.2, 1.5, 1.6], [3.6, 666, 6.7]]

def test_fillna_listarray_array():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    starts  = awkward1.layout.Index64(numpy.array([0, 3, 4, 5, 8]))
    stops   = awkward1.layout.Index64(numpy.array([3, 3, 6, 8, 9]))
    listarray   = awkward1.layout.ListArray64(starts, stops, content)

    assert awkward1.to_list(listarray) == [[0.0, 1.1, 2.2], [], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8]]
    value = awkward1.layout.NumpyArray(numpy.array([55]))
    assert awkward1.to_list(listarray.fillna(value)) == [[0.0, 1.1, 2.2], [], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8]]

def test_fillna_unionarray():
    content1 = awkward1.from_iter([[], [1.1], [2.2, 2.2]], highlevel=False)
    content2 = awkward1.from_iter([[2, 2], [1], []], highlevel=False)
    tags = awkward1.layout.Index8(numpy.array([0, 1, 0, 1, 0, 1], dtype=numpy.int8))
    index = awkward1.layout.Index64(numpy.array([0, 0, 1, 1, 2, 2], dtype=numpy.int64))
    array = awkward1.layout.UnionArray8_64(tags, index, [content1, content2])
    assert awkward1.to_list(array) == [[], [2, 2], [1.1], [1], [2.2, 2.2], []]

    padded_array = array.rpad(2, 1)
    assert awkward1.to_list(padded_array) == [[None, None], [2, 2], [1.1, None], [1, None], [2.2, 2.2], [None, None]]
    value = awkward1.layout.NumpyArray(numpy.array([777]))
    assert awkward1.to_list(padded_array.fillna(value)) == [[777, 777], [2, 2], [1.1, 777], [1, 777], [2.2, 2.2], [777, 777]]

def test_highlevel():
    array = awkward1.Array([[1.1, 2.2, None, 3.3], [], [4.4, None, 5.5]])
    assert awkward1.to_list(awkward1.fill_none(array, 999)) == [[1.1, 2.2, 999, 3.3], [], [4.4, 999, 5.5]]
    assert awkward1.to_list(awkward1.fill_none(array, [1, 2, 3])) == [[1.1, 2.2, [1, 2, 3], 3.3], [], [4.4, [1, 2, 3], 5.5]]
    assert awkward1.to_list(awkward1.fill_none(array, [])) == [[1.1, 2.2, [], 3.3], [], [4.4, [], 5.5]]
    assert awkward1.to_list(awkward1.fill_none(array, {"x": 999})) == [[1.1, 2.2, {"x": 999}, 3.3], [], [4.4, {"x": 999}, 5.5]]

    array = awkward1.Array([[1.1, 2.2, 3.3], None, [], None, [4.4, 5.5]])
    assert awkward1.to_list(awkward1.fill_none(array, 999)) == [[1.1, 2.2, 3.3], 999, [], 999, [4.4, 5.5]]
    assert awkward1.to_list(awkward1.fill_none(array, [1, 2, 3])) == [[1.1, 2.2, 3.3], [1, 2, 3], [], [1, 2, 3], [4.4, 5.5]]
    assert awkward1.to_list(awkward1.fill_none(array, {"x": 999})) == [[1.1, 2.2, 3.3], {"x": 999}, [], {"x": 999}, [4.4, 5.5]]
    assert awkward1.to_list(awkward1.fill_none(array, [])) == [[1.1, 2.2, 3.3], [], [], [], [4.4, 5.5]]
