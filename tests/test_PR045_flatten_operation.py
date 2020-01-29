# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def flatten(data, axis=0):
    if axis < 0:
        raise NotImplementedError("axis < 0 is much harder for untyped data...")
    if isinstance(data, list):
        if axis == 0:
            if all(isinstance(x, list) for x in data):
                return sum(data, [])
            else:
                raise ValueError("cannot concatenate non-lists")
        else:
            return [flatten(x, axis - 1) for x in data]
    elif isinstance(data, dict):
        return {n: flatten(x, axis) for n, x in data.items()}   # does not reduce axis!
    else:
        raise ValueError("cannot flatten {0} objects".format(type(data)))

def test_flatten_empty_array():
    empty = awkward1.layout.EmptyArray()

    assert awkward1.tolist(empty) == []
    assert awkward1.tolist(empty.flatten()) == []

def test_flatten_list_array():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    starts  = awkward1.layout.Index64(numpy.array([0, 3, 4, 5, 8]))
    stops   = awkward1.layout.Index64(numpy.array([3, 3, 6, 8, 9]))
    array   = awkward1.layout.ListArray64(starts, stops, content)

    assert awkward1.tolist(array) == [[0.0, 1.1, 2.2], [], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8]]
    assert awkward1.tolist(array.flatten()) == [0.0, 1.1, 2.2, 4.4, 5.5, 5.5, 6.6, 7.7, 8.8]
    assert awkward1.tolist(array.flatten(-1)) == [0.0, 1.1, 2.2, 4.4, 5.5, 5.5, 6.6, 7.7, 8.8]

    array2 = array[2:-1]
    assert awkward1.tolist(array2) == [[4.4, 5.5], [5.5, 6.6, 7.7]]
    assert flatten(awkward1.tolist(array2)) == [4.4, 5.5, 5.5, 6.6, 7.7]
    assert awkward1.tolist(array2.flatten()) == [4.4, 5.5, 5.5, 6.6, 7.7]

    # The following are allowed:
    #     * out of order (4:7 before 0:1)
    #     * overlaps (0:1 and 0:4 and 1:5)
    #     * beyond content starts[i] == stops[i] (999)
    #
    # See https://github.com/scikit-hep/awkward-1.0/wiki/ListArray.md
    #
    content = awkward1.layout.NumpyArray(numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=numpy.int64))
    starts  = awkward1.layout.Index64(numpy.array([4, 999, 0, 0, 1, 7]))
    stops   = awkward1.layout.Index64(numpy.array([7, 999, 1, 4, 5, 10]))
    array   = awkward1.layout.ListArray64(starts, stops, content)
    assert awkward1.tolist(array) == [[4, 5, 6], [], [0], [0, 1, 2, 3], [1, 2, 3, 4], [7, 8, 9]]
    assert flatten(awkward1.tolist(array)) == [4, 5, 6, 0, 0, 1, 2, 3, 1, 2, 3, 4, 7, 8, 9]
    assert awkward1.tolist(array.flatten()) == [4, 5, 6, 0, 0, 1, 2, 3, 1, 2, 3, 4, 7, 8, 9]

def test_flatten_list_offset_array():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10], dtype=numpy.int64))
    array   = awkward1.layout.ListOffsetArray64(offsets, content)

    assert awkward1.tolist(array) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.tolist(array.flatten()) == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    assert awkward1.tolist(array.flatten(-1)) == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

    # ValueError: ListOffsetArrayOf<T> cannot be flattened in axis -2 because its depth is 2
    # assert awkward1.tolist(array.flatten(-2)) == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

    array2 = array[2:-1]
    assert awkward1.tolist(array2.flatten()) == [3.3, 4.4, 5.5]

def test_flatten_numpy_array():
    array = awkward1.layout.NumpyArray(numpy.arange(2*3*5, dtype=numpy.int64).reshape(2, 3, 5))
    assert awkward1.tolist(array) == [[[ 0,  1,  2,  3,  4], [ 5,  6,  7,  8,  9], [10, 11, 12, 13, 14]],
                                      [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]
    assert flatten(awkward1.tolist(array)) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
    assert awkward1.tolist(array.flatten()) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
    assert flatten(awkward1.tolist(array), 1) == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]
    assert awkward1.tolist(array.flatten(1)) == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]
    assert awkward1.tolist(array.flatten(-1)) == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]
    assert awkward1.tolist(array.flatten(-2)) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]

    array2 = awkward1.layout.NumpyArray(numpy.arange(1*6*5, dtype=numpy.int64).reshape(1, 6, 5))
    assert awkward1.tolist(array2.flatten()) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
    assert awkward1.tolist(array2.flatten(1)) == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]

    # ValueError: NumpyArray cannot be flattened because axis is 2 exeeds its 3 dimensions
    # assert awkward1.tolist(array2.flatten(2)) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

    array3 = array[:, 1::2, :3]
    print(awkward1.tolist(array3))
    assert awkward1.tolist(array3) == [[[5, 6, 7]], [[20, 21, 22]]]
    assert flatten(awkward1.tolist(array3)) == [[5, 6, 7], [20, 21, 22]]
    assert awkward1.tolist(array3.flatten()) == [[5, 6, 7], [20, 21, 22]]

    another_arr = awkward1.layout.NumpyArray(numpy.arange(2*3*5, dtype=numpy.int64).reshape(-1, array.shape[-1]))
    assert awkward1.tolist(another_arr) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
    assert awkward1.tolist(another_arr.flatten()) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

def test_fail_flatten_numpy_array():
    # Fail if axis == shape.size()
    array = awkward1.layout.NumpyArray(numpy.arange(2*3*5, dtype=numpy.int64).reshape(2, 3, 5))

    # The following produces a ValueError: cannot concatenate non-lists
    with pytest.raises(ValueError, match="cannot concatenate non-lists") :
        assert flatten(awkward1.tolist(array), 2) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    # Cannot flatten if axes >= shape.size() - 1:
    with pytest.raises(ValueError) :
        assert awkward1.tolist(array.flatten(2)) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

## def test_flatten_raw_array():
    ## RawArrayOf<T> is usable only in C++

def test_flatten_record():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10], dtype=numpy.int64))
    #FIXME: array   = awkward1.layout.Record()

def test_flatten_record_array():
    array = awkward1.layout.NumpyArray(numpy.arange(2*3*5, dtype=numpy.int64).reshape(2, 3, 5))

    #with pytest.raises(ValueError, match="NumpyArray cannot be flattened because it has 1 dimensions"):
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content2)
    recordarray = awkward1.layout.RecordArray({"one": content1, "two": listoffsetarray})
    recordarray1 = awkward1.layout.RecordArray({"one": array, "two": listoffsetarray})

    assert awkward1.tolist(recordarray) == [{'one': 1, 'two': [1.1, 2.2, 3.3]}, {'one': 2, 'two': []}, {'one': 3, 'two': [4.4, 5.5]}, {'one': 4, 'two': [6.6]}, {'one': 5, 'two': [7.7, 8.8, 9.9]}]
    #assert flatten(awkward1.tolist(recordarray)) == [{'one': [1, 2, 3, 4, 5], 'two': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]}]
    #assert awkward1.tolist(recordarray.flatten()) == [{'one': [1, 2, 3, 4, 5], 'two': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]}]
    assert awkward1.tolist(recordarray1) == [{'one': [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]], 'two': [1.1, 2.2, 3.3]}, {'one': [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]], 'two': []}]
    # FIXME: assert flatten(awkward1.tolist(recordarray1)) == [{'one': [1, 2, 3, 4, 5], 'two': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]}]
    assert awkward1.tolist(recordarray1.flatten()) == [{'one': [0, 1, 2, 3, 4], 'two': 1.1}, {'one': [5, 6, 7, 8, 9], 'two': 2.2}, {'one': [10, 11, 12, 13, 14], 'two': 3.3}, {'one': [15, 16, 17, 18, 19], 'two': 4.4}, {'one': [20, 21, 22, 23, 24], 'two': 5.5}, {'one': [25, 26, 27, 28, 29], 'two': 6.6}]
    ##assert awkward1.tolist(recordarray.flatten()) == [{'one': [1, 2, 3, 4, 5], 'two': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]}]

def test_flatten_regular_array():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
    assert awkward1.tolist(listoffsetarray) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9], []]
    regulararray = awkward1.layout.RegularArray(listoffsetarray, 2)
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 6, 10]))
    listoffsetarray2 = awkward1.layout.ListOffsetArray64(offsets2, content)
    assert awkward1.tolist(listoffsetarray2) == [[0.0, 1.1, 2.2], [3.3, 4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]
    regulararray2 = awkward1.layout.RegularArray(listoffsetarray2, 1)
    assert awkward1.tolist(regulararray2) == [[[0.0, 1.1, 2.2]], [[3.3, 4.4, 5.5]], [[6.6, 7.7, 8.8, 9.9]]]

    assert awkward1.tolist(regulararray) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]
    assert flatten(awkward1.tolist(regulararray)) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9], []]
    assert awkward1.tolist(regulararray.flatten()) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9], []]
    assert flatten(awkward1.tolist(regulararray), 1) == [[0.0, 1.1, 2.2], [3.3, 4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.tolist(regulararray.flatten(1)) == [[0.0, 1.1, 2.2], [3.3, 4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.tolist(regulararray.flatten(-1)) == [[0.0, 1.1, 2.2], [3.3, 4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.tolist(regulararray.flatten(-2)) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9], []]

    # ValueError: RegularArray cannot be flattened in axis -3 because its depth is 3
    # assert awkward1.tolist(regulararray.flatten(-3)) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9], []]

    # cannot concatenate non-lists
    # assert flatten(awkward1.tolist(regulararray), 2) == [[0.0, 1.1, 2.2], [3.3, 4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]

    # NumpyArray cannot be flattened because it has 1 dimensions:
    # assert awkward1.tolist(regulararray.flatten(2)) == [[0.0, 1.1, 2.2], [3.3, 4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]

    regulararray_f1 = awkward1.layout.RegularArray(listoffsetarray, 3)
    assert awkward1.tolist(regulararray_f1) == [[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [[5.5], [6.6, 7.7, 8.8, 9.9], []]]

    starts = awkward1.layout.Index64(numpy.array([0, 1]))
    stops = awkward1.layout.Index64(numpy.array([2, 3]))
    listarray = awkward1.layout.ListArray64(starts, stops, regulararray)
    assert awkward1.tolist(listarray) == [[[[0.0, 1.1, 2.2], []],    [[3.3, 4.4],           [5.5]]],
                                          [[[3.3, 4.4],      [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]]
    assert flatten(awkward1.tolist(listarray)) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]
    assert awkward1.tolist(listarray.flatten()) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]
    assert flatten(awkward1.tolist(listarray), 1) == [[[0.0, 1.1, 2.2], [],    [3.3, 4.4],           [5.5]],
                                                      [[3.3, 4.4],      [5.5], [6.6, 7.7, 8.8, 9.9], []]]
    assert awkward1.tolist(listarray.flatten(1)) == [[[0.0, 1.1, 2.2], [],    [3.3, 4.4],           [5.5]],
                                                      [[3.3, 4.4],      [5.5], [6.6, 7.7, 8.8, 9.9], []]]
    assert flatten(awkward1.tolist(listarray), 2) == [[[0.0, 1.1, 2.2], [3.3, 4.4, 5.5]],
                                                      [[3.3, 4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]]
    assert awkward1.tolist(listarray.flatten(2)) == [[[0.0, 1.1, 2.2], [3.3, 4.4, 5.5]],
                                                      [[3.3, 4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]]
    assert awkward1.tolist(listarray.flatten(-1)) == [[[0.0, 1.1, 2.2], [3.3, 4.4, 5.5]],
                                                      [[3.3, 4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]]
    assert awkward1.tolist(listarray.flatten(-2)) == [[[0.0, 1.1, 2.2], [],    [3.3, 4.4],           [5.5]],
                                                      [[3.3, 4.4],      [5.5], [6.6, 7.7, 8.8, 9.9], []]]
    assert awkward1.tolist(listarray.flatten(-3)) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]

    # ValueError: ListArrayOf<T> cannot be flattened in axis -4 because its depth is 4
    # assert awkward1.tolist(listarray.flatten(-4)) == []

    regulararray_m1 = awkward1.layout.RegularArray(listoffsetarray[:-1], 2)

    assert awkward1.tolist(regulararray_m1) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]]]
    assert flatten(awkward1.tolist(regulararray_m1)) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5]]
    assert awkward1.tolist(regulararray_m1.flatten()) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5]]
    assert flatten(awkward1.tolist(regulararray_m1), 1) == [[0.0, 1.1, 2.2], [3.3, 4.4, 5.5]]
    assert awkward1.tolist(regulararray_m1.flatten(1)) == [[0.0, 1.1, 2.2], [3.3, 4.4, 5.5]]

    regulararray_m2 = awkward1.layout.RegularArray(listoffsetarray[:-2], 2)

    assert awkward1.tolist(regulararray_m2) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]]]
    assert flatten(awkward1.tolist(regulararray_m2)) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5]]
    assert awkward1.tolist(regulararray_m2.flatten()) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5]]

    starts  = awkward1.layout.Index64(numpy.array([0, 3, 4, 5, 8]))
    stops   = awkward1.layout.Index64(numpy.array([3, 3, 6, 8, 9]))
    listarray   = awkward1.layout.ListArray64(starts, stops, content)
    regulararray_m3 = awkward1.layout.RegularArray(listarray, 1)

    assert awkward1.tolist(regulararray_m3) == [[[0.0, 1.1, 2.2]], [[]], [[4.4, 5.5]], [[5.5, 6.6, 7.7]], [[8.8]]]
    assert flatten(awkward1.tolist(regulararray_m3)) == [[0.0, 1.1, 2.2], [], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8]]
    assert awkward1.tolist(regulararray_m3.flatten()) == [[0.0, 1.1, 2.2], [], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8]]
    assert flatten(awkward1.tolist(regulararray_m3), 1) == [[0.0, 1.1, 2.2], [], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8]]
    assert awkward1.tolist(regulararray_m3.flatten(1)) == [[0.0, 1.1, 2.2], [], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8]]
    assert awkward1.tolist(regulararray_m3.flatten(-1)) == [[0.0, 1.1, 2.2], [], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8]]
