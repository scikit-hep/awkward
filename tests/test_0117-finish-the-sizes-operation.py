# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_emptyarray():
    array = awkward1.layout.EmptyArray()
    assert awkward1.to_list(array.num(0)) == 0
    assert awkward1.to_list(array.num(1)) == []
    assert awkward1.to_list(array.num(2)) == []

def test_numpyarray():
    array = awkward1.layout.NumpyArray(numpy.arange(2*3*5*7).reshape(2, 3, 5, 7))
    assert array.num(0) == 2
    assert awkward1.to_list(array.num(1)) == [3, 3]
    assert awkward1.to_list(array.num(2)) == [[5, 5, 5], [5, 5, 5]]
    assert awkward1.to_list(array.num(3)) == [[[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]], [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]]]
    with pytest.raises(ValueError) as err:
        array.num(4)
    assert str(err.value).startswith("'axis' out of range for 'num'")

def test_regulararray():
    array = awkward1.layout.NumpyArray(numpy.arange(2*3*5*7).reshape(2, 3, 5, 7)).toRegularArray()
    assert array.num(0) == 2
    assert awkward1.to_list(array.num(1)) == [3, 3]
    assert awkward1.to_list(array.num(2)) == [[5, 5, 5], [5, 5, 5]]
    assert awkward1.to_list(array.num(3)) == [[[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]], [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]]]
    with pytest.raises(ValueError) as err:
        array.num(4)
    assert str(err.value).startswith("'axis' out of range for 'num'")

def test_listarray():
    content = awkward1.layout.NumpyArray(numpy.arange(2*3*5).reshape(5, 3, 2))
    starts = awkward1.layout.Index64(numpy.array([0, 3, 3], dtype=numpy.int64))
    stops = awkward1.layout.Index64(numpy.array([3, 3, 5], dtype=numpy.int64))
    array = awkward1.layout.ListArray64(starts, stops, content)
    assert awkward1.to_list(array) == [
        [[[ 0,  1], [ 2,  3], [ 4,  5]],
         [[ 6,  7], [ 8,  9], [10, 11]],
         [[12, 13], [14, 15], [16, 17]]],
        [],
        [[[18, 19], [20, 21], [22, 23]],
         [[24, 25], [26, 27], [28, 29]]]]

    assert awkward1.to_list(array.num(0)) == 3
    assert awkward1.to_list(array.num(1)) == [3, 0, 2]
    assert awkward1.to_list(array.num(2)) == [[3, 3, 3], [], [3, 3]]
    assert awkward1.to_list(array.num(3)) == [[[2, 2, 2], [2, 2, 2], [2, 2, 2]], [], [[2, 2, 2], [2, 2, 2]]]
    with pytest.raises(ValueError) as err:
        array.num(4)
    assert str(err.value).startswith("'axis' out of range for 'num'")

def test_listoffsetarray():
    content = awkward1.layout.NumpyArray(numpy.arange(2*3*5).reshape(5, 3, 2))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64))
    array = awkward1.layout.ListOffsetArray64(offsets, content)
    assert awkward1.to_list(array) == [
        [[[ 0,  1], [ 2,  3], [ 4,  5]],
         [[ 6,  7], [ 8,  9], [10, 11]],
         [[12, 13], [14, 15], [16, 17]]],
        [],
        [[[18, 19], [20, 21], [22, 23]],
         [[24, 25], [26, 27], [28, 29]]]]

    assert awkward1.to_list(array.num(0)) == 3
    assert awkward1.to_list(array.num(1)) == [3, 0, 2]
    assert awkward1.to_list(array.num(2)) == [[3, 3, 3], [], [3, 3]]
    assert awkward1.to_list(array.num(3)) == [[[2, 2, 2], [2, 2, 2], [2, 2, 2]], [], [[2, 2, 2], [2, 2, 2]]]
    with pytest.raises(ValueError) as err:
        array.num(4)
    assert str(err.value).startswith("'axis' out of range for 'num'")

def test_indexedarray():
    content = awkward1.layout.NumpyArray(numpy.arange(2*3*5).reshape(5, 3, 2))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64))
    listarray = awkward1.layout.ListOffsetArray64(offsets, content)
    index = awkward1.layout.Index64(numpy.array([2, 2, 1, 0], dtype=numpy.int64))
    array = awkward1.layout.IndexedArray64(index, listarray)
    assert awkward1.to_list(array) == [
        [[[18, 19], [20, 21], [22, 23]],
         [[24, 25], [26, 27], [28, 29]]],
        [[[18, 19], [20, 21], [22, 23]],
         [[24, 25], [26, 27], [28, 29]]],
        [],
        [[[ 0,  1], [ 2,  3], [ 4,  5]],
         [[ 6,  7], [ 8,  9], [10, 11]],
         [[12, 13], [14, 15], [16, 17]]]]

    assert awkward1.to_list(array.num(0)) == 4
    assert awkward1.to_list(array.num(1)) == [2, 2, 0, 3]
    assert awkward1.to_list(array.num(2)) == [[3, 3], [3, 3], [], [3, 3, 3]]
    assert awkward1.to_list(array.num(3)) == [[[2, 2, 2], [2, 2, 2]], [[2, 2, 2], [2, 2, 2]], [], [[2, 2, 2], [2, 2, 2], [2, 2, 2]]]

    with pytest.raises(ValueError) as err:
        array.num(4)
    assert str(err.value).startswith("'axis' out of range for 'num'")

def test_indexedoptionarray():
    content = awkward1.layout.NumpyArray(numpy.arange(2*3*5).reshape(5, 3, 2))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64))
    listarray = awkward1.layout.ListOffsetArray64(offsets, content)
    index = awkward1.layout.Index64(numpy.array([2, -1, 2, 1, -1, 0], dtype=numpy.int64))
    array = awkward1.layout.IndexedOptionArray64(index, listarray)
    assert awkward1.to_list(array) == [
        [[[18, 19], [20, 21], [22, 23]],
         [[24, 25], [26, 27], [28, 29]]],
        None,
        [[[18, 19], [20, 21], [22, 23]],
         [[24, 25], [26, 27], [28, 29]]],
        [],
        None,
        [[[ 0,  1], [ 2,  3], [ 4,  5]],
         [[ 6,  7], [ 8,  9], [10, 11]],
         [[12, 13], [14, 15], [16, 17]]]]

    assert awkward1.to_list(array.num(0)) == 6
    assert awkward1.to_list(array.num(1)) == [2, None, 2, 0, None, 3]
    assert awkward1.to_list(array.num(2)) == [[3, 3], None, [3, 3], [], None, [3, 3, 3]]
    assert awkward1.to_list(array.num(3)) == [[[2, 2, 2], [2, 2, 2]], None, [[2, 2, 2], [2, 2, 2]], [], None, [[2, 2, 2], [2, 2, 2], [2, 2, 2]]]

    with pytest.raises(ValueError) as err:
        array.num(4)
    assert str(err.value).startswith("'axis' out of range for 'num'")

def test_recordarray():
    array = awkward1.from_iter([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}], highlevel=False)
    assert awkward1.to_list(array.num(0)) == {"x": 4, "y": 4}

    array = awkward1.from_iter([{"x": [3.3, 3.3, 3.3], "y": []}, {"x": [2.2, 2.2], "y": [1]}, {"x": [1.1], "y": [2, 2]}, {"x": [], "y": [3, 3, 3]}], highlevel=False)
    assert awkward1.to_list(array.num(0)) == {"x": 4, "y": 4}
    assert awkward1.to_list(array.num(1)) == [{"x": 3, "y": 0}, {"x": 2, "y": 1}, {"x": 1, "y": 2}, {"x": 0, "y": 3}]
    assert awkward1.to_list(array.num(1)[2]) == {"x": 1, "y": 2}

    array = awkward1.from_iter([{"x": [[3.3, 3.3, 3.3]], "y": []}, {"x": [[2.2, 2.2]], "y": [1]}, {"x": [[1.1]], "y": [2, 2]}, {"x": [[]], "y": [3, 3, 3]}], highlevel=False)
    assert awkward1.to_list(array.num(0)) == {"x": 4, "y": 4}
    assert awkward1.to_list(array.num(1)) == [{"x": 1, "y": 0}, {"x": 1, "y": 1}, {"x": 1, "y": 2}, {"x": 1, "y": 3}]
    assert awkward1.to_list(array.num(1)[2]) == {"x": 1, "y": 2}

def test_unionarray():
    content1 = awkward1.from_iter([[], [1], [2, 2], [3, 3, 3]], highlevel=False)
    content2 = awkward1.from_iter([[3.3, 3.3, 3.3], [2.2, 2.2], [1.1], []], highlevel=False)
    tags = awkward1.layout.Index8(numpy.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=numpy.int8))
    index = awkward1.layout.Index64(numpy.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=numpy.int64))
    array = awkward1.layout.UnionArray8_64(tags, index, [content1, content2])
    assert awkward1.to_list(array) == [[], [3.3, 3.3, 3.3], [1], [2.2, 2.2], [2, 2], [1.1], [3, 3, 3], []]

    assert array.num(0) == 8
    assert isinstance(array.num(1), awkward1.layout.NumpyArray)
    assert awkward1.to_list(array.num(1)) == [0, 3, 1, 2, 2, 1, 3, 0]

def test_highlevel():
    array = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    assert awkward1.to_list(awkward1.num(array)) == [3, 0, 2]

def test_flatten_ListOffsetArray():
    array = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    assert awkward1.to_list(awkward1.flatten(array)) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.to_list(awkward1.flatten(array[1:])) == [4.4, 5.5]

    array = awkward1.Array([[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]], [[], [6.6, 7.7, 8.8, 9.9]]])
    assert awkward1.to_list(awkward1.flatten(array)) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.to_list(awkward1.flatten(array[1:])) == [[5.5], [], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.to_list(awkward1.flatten(array[:, 1:])) == [[], [3.3, 4.4], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.to_list(awkward1.flatten(array, axis=2)) == [[0.0, 1.1, 2.2, 3.3, 4.4], [], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.to_list(awkward1.flatten(array[1:], axis=2)) == [[], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.to_list(awkward1.flatten(array[:, 1:], axis=2)) == [[3.3, 4.4], [], [], [6.6, 7.7, 8.8, 9.9]]

    array = awkward1.Array(numpy.arange(2*3*5*7).reshape(2, 3, 5, 7).tolist())
    assert awkward1.to_list(awkward1.flatten(array, axis=1)) == numpy.arange(2*3*5*7).reshape(2 * 3, 5, 7).tolist()
    assert awkward1.to_list(awkward1.flatten(array, axis=2)) == numpy.arange(2*3*5*7).reshape(2, 3 * 5, 7).tolist()
    assert awkward1.to_list(awkward1.flatten(array, axis=3)) == numpy.arange(2*3*5*7).reshape(2, 3, 5 * 7).tolist()

    def toListArray(x):
        if isinstance(x, awkward1.layout.ListOffsetArray64):
            starts = awkward1.layout.Index64(numpy.asarray(x.offsets)[:-1])
            stops  = awkward1.layout.Index64(numpy.asarray(x.offsets)[1:])
            return awkward1.layout.ListArray64(starts, stops, toListArray(x.content))
        elif isinstance(x, awkward1.layout.ListOffsetArray32):
            starts = awkward1.layout.Index64(numpy.asarray(x.offsets)[:-1])
            stops  = awkward1.layout.Index64(numpy.asarray(x.offsets)[1:])
            return awkward1.layout.ListArray32(starts, stops, toListArray(x.content))
        elif isinstance(x, awkward1.layout.ListOffsetArrayU32):
            starts = awkward1.layout.Index64(numpy.asarray(x.offsets)[:-1])
            stops  = awkward1.layout.Index64(numpy.asarray(x.offsets)[1:])
            return awkward1.layout.ListArrayU32(starts, stops, toListArray(x.content))
        else:
            return x

    array = awkward1.Array(toListArray(awkward1.from_iter(numpy.arange(2*3*5*7).reshape(2, 3, 5, 7).tolist(), highlevel=False)))
    assert awkward1.to_list(awkward1.flatten(array, axis=1)) == numpy.arange(2*3*5*7).reshape(2 * 3, 5, 7).tolist()
    assert awkward1.to_list(awkward1.flatten(array, axis=2)) == numpy.arange(2*3*5*7).reshape(2, 3 * 5, 7).tolist()
    assert awkward1.to_list(awkward1.flatten(array, axis=3)) == numpy.arange(2*3*5*7).reshape(2, 3, 5 * 7).tolist()

    array = awkward1.Array(numpy.arange(2*3*5*7).reshape(2, 3, 5, 7))
    assert awkward1.to_list(awkward1.flatten(array, axis=1)) == numpy.arange(2*3*5*7).reshape(2 * 3, 5, 7).tolist()
    assert awkward1.to_list(awkward1.flatten(array, axis=2)) == numpy.arange(2*3*5*7).reshape(2, 3 * 5, 7).tolist()
    assert awkward1.to_list(awkward1.flatten(array, axis=3)) == numpy.arange(2*3*5*7).reshape(2, 3, 5 * 7).tolist()

def test_flatten_IndexedArray():
    array = awkward1.Array([[1.1, 2.2, None, 3.3], None, [], None, [4.4, 5.5], None])
    assert awkward1.to_list(awkward1.flatten(array)) == [1.1, 2.2, None, 3.3, 4.4, 5.5]
    assert awkward1.to_list(awkward1.flatten(array[1:])) == [4.4, 5.5]

    array = awkward1.Array([[[0.0, 1.1, 2.2], None, None, [3.3, 4.4]], [], [[5.5]], [[], [6.6, 7.7, 8.8, 9.9]]])
    assert awkward1.to_list(awkward1.flatten(array, axis=2)) == [[0.0, 1.1, 2.2, 3.3, 4.4], [], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.to_list(awkward1.flatten(array[1:], axis=2)) == [[], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.to_list(awkward1.flatten(array[:, 1:], axis=2)) == [[3.3, 4.4], [], [], [6.6, 7.7, 8.8, 9.9]]

    array = awkward1.Array([[[0.0, 1.1, 2.2], [3.3, 4.4]], [], [[5.5]], None, None, [[], [6.6, 7.7, 8.8, 9.9]]])
    assert awkward1.to_list(awkward1.flatten(array, axis=2)) == [[0.0, 1.1, 2.2, 3.3, 4.4], [], [5.5], None, None, [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.to_list(awkward1.flatten(array[1:], axis=2)) == [[], [5.5], None, None, [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.to_list(awkward1.flatten(array[:, 1:], axis=2)) == [[3.3, 4.4], [], [], None, None, [6.6, 7.7, 8.8, 9.9]]

    array = awkward1.Array([[[0.0, 1.1, None, 2.2], None, [], None, [3.3, 4.4]], None, [], [[5.5]], None, [[], [6.6, None, 7.7, 8.8, 9.9], None]])
    assert awkward1.to_list(awkward1.flatten(array)) == [[0.0, 1.1, None, 2.2], None, [], None, [3.3, 4.4], [5.5], [], [6.6, None, 7.7, 8.8, 9.9], None]
    assert awkward1.to_list(awkward1.flatten(array, axis=2)) == [[0.0, 1.1, None, 2.2, 3.3, 4.4], None, [], [5.5], None, [6.6, None, 7.7, 8.8, 9.9]]
    assert awkward1.to_list(awkward1.flatten(array[1:], axis=2)) == [None, [], [5.5], None, [6.6, None, 7.7, 8.8, 9.9]]
    assert awkward1.to_list(awkward1.flatten(array[:, 1:], axis=2)) == [[3.3, 4.4], None, [], [], None, [6.6, None, 7.7, 8.8, 9.9]]

    content = awkward1.from_iter([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False)
    index = awkward1.layout.Index64(numpy.array([2, 1, 0, 3, 3, 4], dtype=numpy.int64))
    array = awkward1.Array(awkward1.layout.IndexedArray64(index, content))
    assert awkward1.to_list(array) == [[3.3, 4.4], [], [0.0, 1.1, 2.2], [5.5], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.to_list(awkward1.flatten(array)) == [3.3, 4.4, 0.0, 1.1, 2.2, 5.5, 5.5, 6.6, 7.7, 8.8, 9.9]

    content = awkward1.from_iter([[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]], [[], [6.6, 7.7, 8.8, 9.9]]], highlevel=False)
    index = awkward1.layout.Index64(numpy.array([2, 2, 1, 0, 3], dtype=numpy.int64))
    array = awkward1.Array(awkward1.layout.IndexedArray64(index, content))
    assert awkward1.to_list(array) == [[[5.5]], [[5.5]], [], [[0.0, 1.1, 2.2], [], [3.3, 4.4]], [[], [6.6, 7.7, 8.8, 9.9]]]
    assert awkward1.to_list(awkward1.flatten(array, axis=2)) == [[5.5], [5.5], [], [0.0, 1.1, 2.2, 3.3, 4.4], [6.6, 7.7, 8.8, 9.9]]

def test_flatten_RecordArray():
    array = awkward1.Array([{"x": [], "y": [[3, 3, 3]]}, {"x": [[1]], "y": [[2, 2]]}, {"x": [[2], [2]], "y": [[1]]}, {"x": [[3], [3], [3]], "y": [[]]}])
    assert awkward1.to_list(awkward1.flatten(array, axis=2)) == [{"x": [], "y": [3, 3, 3]}, {"x": [1], "y": [2, 2]}, {"x": [2, 2], "y": [1]}, {"x": [3, 3, 3], "y": []}]
    assert awkward1.to_list(awkward1.flatten(array[1:], axis=2)) == [{"x": [1], "y": [2, 2]}, {"x": [2, 2], "y": [1]}, {"x": [3, 3, 3], "y": []}]
    assert awkward1.to_list(awkward1.flatten(array[:, 1:], axis=2)) == [{"x": [], "y": []}, {"x": [], "y": []}, {"x": [2], "y": []}, {"x": [3, 3], "y": []}]

def test_flatten_UnionArray():
    content1 = awkward1.from_iter([[1.1], [2.2, 2.2], [3.3, 3.3, 3.3]], highlevel=False)
    content2 = awkward1.from_iter([[[3, 3, 3], [3, 3, 3], [3, 3, 3]], [[2, 2], [2, 2]], [[1]]], highlevel=False)
    tags = awkward1.layout.Index8(numpy.array([0, 1, 0, 1, 0, 1], dtype=numpy.int8))
    index= awkward1.layout.Index64(numpy.array([0, 0, 1, 1, 2, 2], dtype=numpy.int64))
    array = awkward1.Array(awkward1.layout.UnionArray8_64(tags, index, [content1, content2]))
    assert awkward1.to_list(array) == [[1.1], [[3, 3, 3], [3, 3, 3], [3, 3, 3]], [2.2, 2.2], [[2, 2], [2, 2]], [3.3, 3.3, 3.3], [[1]]]
    assert awkward1.to_list(array[1:]) == [[[3, 3, 3], [3, 3, 3], [3, 3, 3]], [2.2, 2.2], [[2, 2], [2, 2]], [3.3, 3.3, 3.3], [[1]]]
    assert awkward1.to_list(awkward1.flatten(array)) == [1.1, [3, 3, 3], [3, 3, 3], [3, 3, 3], 2.2, 2.2, [2, 2], [2, 2], 3.3, 3.3, 3.3, [1]]
    assert awkward1.to_list(awkward1.flatten(array[1:])) == [[3, 3, 3], [3, 3, 3], [3, 3, 3], 2.2, 2.2, [2, 2], [2, 2], 3.3, 3.3, 3.3, [1]]

    array = awkward1.Array(awkward1.layout.UnionArray8_64(tags, index, [content2, content2]))
    assert awkward1.to_list(array) == [[[3, 3, 3], [3, 3, 3], [3, 3, 3]], [[3, 3, 3], [3, 3, 3], [3, 3, 3]], [[2, 2], [2, 2]], [[2, 2], [2, 2]], [[1]], [[1]]]
    assert awkward1.to_list(awkward1.flatten(array, axis=2)) == [[3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3], [2, 2, 2, 2], [2, 2, 2, 2], [1], [1]]
    assert awkward1.to_list(awkward1.flatten(array[1:], axis=2)) == [[3, 3, 3, 3, 3, 3, 3, 3, 3], [2, 2, 2, 2], [2, 2, 2, 2], [1], [1]]
    assert awkward1.to_list(awkward1.flatten(array[:, 1:], axis=2)) == [[3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3], [2, 2], [2, 2], [], []]
