# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def count(data, axis=0):
    if axis < 0:
        raise NotImplementedError("axis < 0 is much harder for untyped data...")
    if isinstance(data, list):
        if axis == 0:
            if all(isinstance(x, list) for x in data):
                return [len(x) for x in data]
            else:
                raise ValueError("cannot count the lengths of non-lists")
        else:
            return [count(x, axis - 1) for x in data]
    else:
        raise ValueError("cannot count {0} objects".format(type(data)))

def test_count_empty_array():
    empty = awkward1.layout.EmptyArray()

    assert awkward1.tolist(empty) == []
    assert awkward1.tolist(empty.sizes(0)) == []

def test_count_indexed_array():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10], dtype=numpy.int64))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
    index = awkward1.layout.Index64(numpy.array([2, 0, 1, 3, 3, 4], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray64(index, listoffsetarray)
    assert awkward1.tolist(indexedarray) == [[3.3, 4.4], [0.0, 1.1, 2.2], [], [5.5], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert count(awkward1.tolist(indexedarray), 0) == [2, 3, 0, 1, 1, 4]
    assert awkward1.tolist(indexedarray.sizes(0)) == [2, 3, 0, 1, 1, 4]
    assert awkward1.tolist(indexedarray.sizes(-1)) == [2, 3, 0, 1, 1, 4]

def test_count_list_array():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    starts  = awkward1.layout.Index64(numpy.array([0, 3, 4, 5, 8]))
    stops   = awkward1.layout.Index64(numpy.array([3, 3, 6, 8, 9]))
    array   = awkward1.layout.ListArray64(starts, stops, content)

    assert awkward1.tolist(array) == [[0.0, 1.1, 2.2], [], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8]]
    assert count(awkward1.tolist(array), 0) == [3, 0, 2, 3, 1]
    assert awkward1.tolist(array.sizes(0)) == [3, 0, 2, 3, 1]
    assert awkward1.tolist(array.sizes(-1)) == [3, 0, 2, 3, 1]

    with pytest.raises(ValueError) as err:
        count(awkward1.tolist(array), 1)
    assert str(err.value) == "cannot count the lengths of non-lists"

def test_count_list_offset_array():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10], dtype=numpy.int64))
    array   = awkward1.layout.ListOffsetArray64(offsets, content)

    assert awkward1.tolist(array) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert count(awkward1.tolist(array), 0) == [3, 0, 2, 1, 4]
    assert awkward1.tolist(array.sizes(0)) == [3, 0, 2, 1, 4]
    assert awkward1.tolist(array.sizes(-1)) == [3, 0, 2, 1, 4]

    with pytest.raises(ValueError) as err:
        count(awkward1.tolist(array), 1) == [3, 0, 2, 1, 4]
    assert str(err.value) == "cannot count the lengths of non-lists"

    # FIXME: assert awkward1.tolist(array.sizes(1)) == []

def test_count_numpy_array():
    array = awkward1.layout.NumpyArray(numpy.arange(2*3*5, dtype=numpy.int64).reshape(2, 3, 5))
    assert awkward1.tolist(array) == [[[ 0,  1,  2,  3,  4], [ 5,  6,  7,  8,  9], [10, 11, 12, 13, 14]],
                                      [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]
    assert count(awkward1.tolist(array), 0) == [3, 3]
    assert awkward1.tolist(array.sizes(0)) == [3, 3]
    assert count(awkward1.tolist(array), 1) == [[5, 5, 5], [5, 5, 5]]
    assert awkward1.tolist(array.sizes(1)) == [[5, 5, 5], [5, 5, 5]]
    assert awkward1.tolist(array.sizes(2)) == [30]
    assert awkward1.tolist(array.sizes(-1)) == [[5, 5, 5], [5, 5, 5]]
    assert awkward1.tolist(array.sizes(-2)) == [3, 3]

    array2 = awkward1.layout.NumpyArray(numpy.arange(2*3*5*3, dtype=numpy.int64).reshape(2, 3, 5, 3))
    assert awkward1.tolist(array2) == [[[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]],
                                        [[15, 16, 17], [18, 19, 20], [21, 22, 23], [24, 25, 26], [27, 28, 29]],
                                        [[30, 31, 32], [33, 34, 35], [36, 37, 38], [39, 40, 41], [42, 43, 44]]],
                                       [[[45, 46, 47], [48, 49, 50], [51, 52, 53], [54, 55, 56], [57, 58, 59]],
                                        [[60, 61, 62], [63, 64, 65], [66, 67, 68], [69, 70, 71], [72, 73, 74]],
                                        [[75, 76, 77], [78, 79, 80], [81, 82, 83], [84, 85, 86], [87, 88, 89]]]]
    assert count(awkward1.tolist(array2), 0) == [3, 3]
    assert awkward1.tolist(array2.sizes(0)) == [3, 3]
    assert count(awkward1.tolist(array2), 1) == [[5, 5, 5], [5, 5, 5]]
    assert awkward1.tolist(array2.sizes(1)) == [[5, 5, 5], [5, 5, 5]]
    assert count(awkward1.tolist(array2), 2) == [[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]],
                                                 [[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]]]
    assert awkward1.tolist(array2.sizes(2)) == [[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]],
                                                [[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]]]
    assert awkward1.tolist(array2.sizes(3)) == [90]
    assert awkward1.tolist(array2.sizes(-1)) == [[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]],
                                                [[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]]]
    assert awkward1.tolist(array2.sizes(-2)) == [[5, 5, 5], [5, 5, 5]]
    assert awkward1.tolist(array2.sizes(-3)) == [3, 3]

## def test_count_raw_array():
    ## RawArrayOf<T> is usable only in C++

def test_count_record():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10], dtype=numpy.int64))
    #FIXME: array   = awkward1.layout.Record()

def test_count_record_array():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content2)
    recordarray = awkward1.layout.RecordArray({"one": content1, "two": listoffsetarray})

    assert awkward1.tolist(recordarray) == [{'one': 1, 'two': [1.1, 2.2, 3.3]}, {'one': 2, 'two': []}, {'one': 3, 'two': [4.4, 5.5]}, {'one': 4, 'two': [6.6]}, {'one': 5, 'two': [7.7, 8.8, 9.9]}]

    with pytest.raises(ValueError) as err:
        count(awkward1.tolist(recordarray), 0)
    assert str(err.value) == "cannot count the lengths of non-lists"

    assert(awkward1.tolist(recordarray.sizes(0))) == [{'one': 5, 'two': 3}]

    with pytest.raises(ValueError) as err:
        awkward1.tolist(recordarray.sizes(1))
    assert str(err.value) == "NumpyArray cannot be counted because it has 0 dimensions"

def test_count_regular_array():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
    regulararray = awkward1.layout.RegularArray(listoffsetarray, 2)

    assert awkward1.tolist(regulararray) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]
    assert count(awkward1.tolist(regulararray), 0) == [2, 2, 2]
    assert awkward1.tolist(regulararray.sizes(0)) == [2, 2, 2]
    assert count(awkward1.tolist(regulararray), 1) == [[3, 0], [2, 1], [4, 0]]
    assert awkward1.tolist(regulararray.sizes(1)) == [[3, 0], [2, 1], [4, 0]]
    assert awkward1.tolist(regulararray.sizes(-1)) == [[3, 0], [2, 1], [4, 0]]
    assert awkward1.tolist(regulararray.sizes(-2)) == [2, 2, 2]

def test_count_union_array():
    content0 = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], checkvalid=True).layout
    content1 = awkward1.Array(["one", "two", "three", "four", "five"], checkvalid=True).layout
    tags = awkward1.layout.Index8(numpy.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=numpy.int8))
    index = awkward1.layout.Index32(numpy.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=numpy.int32))
    array = awkward1.layout.UnionArray8_32(tags, index, [content0, content1])
    assert awkward1.tolist(array.sizes(0)) == [3, 3, 3, 0, 5, 2, 4, 4]
    assert awkward1.tolist(array.sizes(-1)) == [3, 3, 3, 0, 5, 2, 4, 4]

    with pytest.raises(ValueError) as err:
        count(awkward1.tolist(array), 0)
    assert str(err.value) == "cannot count the lengths of non-lists"

    ## an example from studies
    offsets3 = awkward1.layout.Index64(numpy.array([25, 29, 57, 69, 88, 90, 98, 99, 115, 141, 173, 178, 238, 248, 250]))
    tags3 = awkward1.layout.Index8(numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=numpy.int8))
    index3 = awkward1.layout.Index32(numpy.array([31, 28, 14, 13, 6, 6, 2, 2, 23, 34, 7, 19, 14, 8, 12,
    6, 34, 17, 25, 5, 28, 28, 17, 12, 1, 12, 9, 28, 3, 3, 3, 27, 11, 27, 0, 16, 5, 11, 18, 5, 17, 0, 3,
    1, 23, 28, 1, 12, 27, 29, 2, 21, 6, 21, 19, 26, 11, 19, 7, 34, 16, 15, 12, 14, 16, 10, 17, 18, 8,
    6, 34, 5, 24, 24, 34, 23, 33, 13, 1, 27, 6, 33, 11, 0, 32, 27, 10, 1, 11, 16, 4, 5, 34, 9, 27, 34,
    32, 13, 21, 28, 4, 6, 8, 3, 26, 0, 16, 5, 8, 4, 25, 0, 29, 22, 19, 18, 32, 23, 3, 27, 32, 5, 28,
    30, 6, 14, 8, 10, 7, 32, 8, 13, 17, 23, 3, 30, 25, 23, 8, 29, 17, 11, 6, 26, 9, 28, 17, 3, 15, 7,
    5, 33, 3, 18, 20, 24, 14, 17, 21, 0, 6, 9, 1, 16, 26, 9, 17, 12, 14, 22, 20, 18, 5, 6, 24, 24, 31,
    25, 17, 7, 32, 31, 16, 25, 1, 2, 1, 13, 21, 25, 8, 34, 33, 5, 24, 20, 27, 30, 13, 6, 30, 16, 14,
    34, 6, 4, 16, 25, 26, 30, 9, 1, 7, 21, 22, 11, 27, 2, 21, 14, 5, 32, 5, 18, 14, 7, 0, 10, 34, 23,
    25, 11, 11, 16, 5, 11, 32, 8, 1, 15, 3, 31, 25, 18, 8, 29, 19, 18, 32, 8, 20, 1, 10], dtype=numpy.int32))
    roffsets3 = awkward1.layout.Index64(numpy.array([2, 28, 29, 47, 55, 65, 80, 86, 92, 92, 94, 107, 111,
    113, 118, 129, 155, 157, 189, 206, 210, 213, 227, 231, 234, 249, 262, 263, 278, 300, 303, 306, 317,
    328, 339, 341], dtype=numpy.int64))
    rcontent3 = awkward1.layout.NumpyArray(numpy.array([5.8, 6.1, -2.5, 5.2, 7.6, 2.1, 10.3, 2.7, 6.4,
    5.4, 8.6, 3.3, 5.8, 3.9, 8.3, 5.6, 5.7, 6.8, 3.7, 0.2, 2.0, 3.8, 2.9, 4.5, 8.2, 7.6, 6.8, 6.8, 5.9,
    9.9, 5.2, 8.4, 5.1, -0.1, -1.8, -4.3, 4.7, 4.6, 0.5, 4.6, 6.2, 6.6, 8.3, 1.6, 1.6, 2.8, 6.6, 4.2, 0.9,
    3.4, 6.1, 7.8, 7.8, 5.2, -1.4, 7.9, 7.4, 5.9, 6.5, 6.5, 8.1, 2.3, 7.1, 10.8, 0.7, 6.1, 2.2, 4.4, 6.8,
    7.9, 7.4, 2.4, 8.8, 3.5, 4.1, 3.8, 7.0, 3.1, 3.1, 8.1, 9.7, 2.5, 4.8, 2.8, 9.7, 5.9, 3.4, 7.6, 3.1,
    4.2, 4.8, 4.2, 0.8, 2.4, 0.6, 8.7, 2.9, 4.0, 4.9, 1.4, 5.3, 6.8, 2.0, 3.3, 2.2, 3.0, 1.3, -3.1, 8.3,
    9.5, 2.7, 10.3, 1.9, 5.6, 5.0, 6.7, 4.9, 4.5, 1.1, 6.6, 8.1, 4.5, 3.8, 3.0, 7.0, 6.2, 1.9, 7.3, 0.5,
    3.3, 7.4, 2.2, 6.0, -0.5, 6.5, 6.4, 6.7, 7.2, 5.8, 2.5, 6.4, 2.3, 6.7, 4.0, 3.5, 6.2, 5.8, 5.7, 7.0,
    3.3, 3.5, 4.3, 8.7, 4.7, -1.2, 7.9, 1.4, 6.4, 2.5, 6.0, 9.8, 5.3, 3.4, 9.3, 2.8, 5.0, 6.3, 6.8, 3.6,
    10.9, 1.6, 0.9, 0.2, 4.8, 7.0, 6.3, 10.4, 8.6, 2.6, 3.5, 4.3, 3.3, 7.3, 4.0, 9.1, 8.6, 2.7, 5.8, 3.5,
    5.1, -1.7, 11.3, 2.8, 1.0, 7.5, 3.6, 5.4, 5.9, 0.5, 6.4, 8.4, -0.5, 7.0, 8.0, 4.5, 1.4, 6.1, 5.5, 1.2,
    4.1, 8.2, 6.4, 2.6, 3.6, 7.1, 0.0, 6.6, 4.2, 5.1, 5.4, 5.5, 2.5, 2.3, 1.2, 9.5, 5.9, 2.2, 1.5, 7.0,
    3.2, 1.6, 4.0, 8.6, 3.0, 9.6, 5.9, 7.2, 6.1, 6.9, 0.6, 6.8, 3.6, 3.9, 2.8, 8.2, 1.1, -1.1, 9.1, 2.2,
    8.9, 3.9, 8.1, 1.5, 3.5, 5.9, 6.5, -0.4, 2.6, 9.4, 7.6, 6.4, 6.1, 11.0, 8.6, 3.5, 5.2, 7.2, 4.0, 11.1,
    3.6, 6.0, 7.9, 7.1, 4.6, 1.9, 3.5, 2.9, 5.9, 4.2, 5.8, 2.4, 5.1, 3.3, 4.8, 0.8, 5.9, 7.8, 7.6, 6.1,
    0.8, 5.1, 5.8, 6.8, 3.5, -0.6, 4.9, 2.1, 2.8, 0.6, 1.4, 7.9, 8.4, 6.6, 1.6, 3.8, 8.9, 0.7, 2.4, 3.9,
    7.3, 5.9, 4.6, 3.0, 5.6, 2.9, 5.9, 5.0, 10.1, 7.3, 0.1, 3.4, 2.4, 7.8, 6.9, 0.0, 4.4, 2.1, 7.7, -1.7,
    1.4, 7.0, 2.4, 6.0, -0.9, 3.8, 7.0, 3.2, 5.1, 3.4, 7.0, 1.0, 4.0, 3.8, -3.3, 6.1, 2.0, 9.0, 4.5, 8.1,
    1.3, 4.1]))
    content03 = awkward1.layout.ListOffsetArray64(roffsets3, rcontent3)
    content3 = awkward1.layout.UnionArray8_32(tags3, index3, [content03])
    array3 = awkward1.layout.ListOffsetArray64(offsets3, content3)
    assert awkward1.tolist(array3.sizes(0)) == [4, 28, 12, 19, 2, 8, 1, 16, 26, 32, 5, 60, 10, 2]
    assert awkward1.tolist(array3.sizes(1)) == [[2, 2, 22, 8],
    [8, 8, 15, 4, 15, 26, 2, 15, 4, 17, 15, 32, 26, 8, 1, 3, 22, 1, 2, 15, 3, 18, 14, 6, 14, 4, 1, 4],
    [4, 6, 2, 2, 26, 2, 11, 2, 13, 32, 17, 0],
    [6, 2, 15, 15, 15, 2, 3, 11, 5, 1, 15, 6, 11, 4, 26, 11, 15, 13, 1],
    [4, 2], [10, 15, 2, 2, 15, 2, 11, 5], [14], [22, 10, 6, 0, 8, 1, 26, 2, 15, 0, 10, 13, 26, 3, 4, 4],
    [17, 11, 3, 8, 15, 11, 15, 22, 3, 6, 11, 0, 13, 6, 11, 0, 5, 32, 3, 8, 3, 13, 3, 0, 3, 32],
    [4, 6, 1, 2, 22, 32, 8, 26, 6, 15, 11, 8, 17, 3, 15, 11, 32, 14, 26, 6, 2, 1, 2, 1, 2, 32, 2, 11, 4, 3, 17, 15],
    [6, 15, 15, 11, 13],
    [32, 6, 11, 11, 2, 13, 1, 18, 1, 5, 14, 13, 0, 2, 11, 15, 15, 3, 15, 3, 5, 6, 3, 2, 11, 2, 6, 10, 2, 13, 1, 3, 2, 1, 6, 14, 4, 4, 15, 18, 14, 11, 15, 11, 15, 17, 11, 6, 26, 13, 2, 3, 13, 4, 4, 2, 15, 4, 11, 0],
    [1, 26, 8, 11, 13, 17, 0, 3, 4, 17], [11, 0]]
