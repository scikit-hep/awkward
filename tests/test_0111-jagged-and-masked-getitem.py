# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak

def test_array_slice():
    array = ak.Array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], check_valid=True)
    assert ak.to_list(array[[5, 2, 2, 3, 9, 0, 1]]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert ak.to_list(array[np.array([5, 2, 2, 3, 9, 0, 1])]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert ak.to_list(array[ak.layout.NumpyArray(np.array([5, 2, 2, 3, 9, 0, 1], dtype=np.int32))]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert ak.to_list(array[ak.Array(np.array([5, 2, 2, 3, 9, 0, 1], dtype=np.int32), check_valid=True)]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert ak.to_list(array[ak.Array([5, 2, 2, 3, 9, 0, 1], check_valid=True)]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]

    assert ak.to_list(array[ak.layout.NumpyArray(np.array([False, False, False, False, False, True, False, True, False, True]))]) == [5.5, 7.7, 9.9]

    content = ak.layout.NumpyArray(np.array([1, 0, 9, 3, 2, 2, 5], dtype=np.int64))
    index = ak.layout.Index64(np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, content)
    assert ak.to_list(array[indexedarray]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert ak.to_list(array[ak.Array(indexedarray, check_valid=True)]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]

    assert ak.to_list(array[ak.layout.EmptyArray()]) == []

    content0 = ak.layout.NumpyArray(np.array([5, 2, 2]))
    content1 = ak.layout.NumpyArray(np.array([3, 9, 0, 1]))
    tags = ak.layout.Index8(np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int8))
    index2 = ak.layout.Index64(np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int64))
    unionarray = ak.layout.UnionArray8_64(tags, index2, [content0, content1])
    assert ak.to_list(array[unionarray]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert ak.to_list(array[ak.Array(unionarray, check_valid=True)]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]

    array = ak.Array(np.array([[0.0, 1.1, 2.2, 3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]), check_valid=True)
    assert ak.to_list(array[ak.layout.NumpyArray(np.array([[0, 1], [1, 0]])), ak.layout.NumpyArray(np.array([[2, 4], [3, 3]]))]) == [[2.2, 9.9], [8.8, 3.3]]
    assert ak.to_list(array[ak.layout.NumpyArray(np.array([[0, 1], [1, 0]]))]) == [[[0.0, 1.1, 2.2, 3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]], [[5.5, 6.6, 7.7, 8.8, 9.9], [0.0, 1.1, 2.2, 3.3, 4.4]]]

    array = ak.Array([{"x": 1, "y": 1.1, "z": [1]}, {"x": 2, "y": 2.2, "z": [2, 2]}, {"x": 3, "y": 3.3, "z": [3, 3, 3]}, {"x": 4, "y": 4.4, "z": [4, 4, 4, 4]}, {"x": 5, "y": 5.5, "z": [5, 5, 5, 5, 5]}], check_valid=True)
    ak.to_list(array[ak.from_iter(["y", "x"], highlevel=False)]) == [{"y": 1.1, "x": 1}, {"y": 2.2, "x": 2}, {"y": 3.3, "x": 3}, {"y": 4.4, "x": 4}, {"y": 5.5, "x": 5}]

def test_new_slices():
    content = ak.layout.NumpyArray(np.array([1, 0, 9, 3, 2, 2, 5], dtype=np.int64))
    index = ak.layout.Index64(np.array([6, 5, -1, 3, 2, -1, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedOptionArray64(index, content)
    assert ak.to_list(indexedarray) == [5, 2, None, 3, 9, None, 1]

    assert ak._ext._slice_tostring(indexedarray) == "[missing([0, 1, -1, ..., 3, -1, 4], array([5, 2, 3, 9, 1]))]"

    offsets = ak.layout.Index64(np.array([0, 4, 4, 7], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)
    assert ak.to_list(listoffsetarray) == [[1, 0, 9, 3], [], [2, 2, 5]]

    assert ak._ext._slice_tostring(listoffsetarray) == "[jagged([0, 4, 4, 7], array([1, 0, 9, ..., 2, 2, 5]))]"

    offsets = ak.layout.Index64(np.array([1, 4, 4, 6], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)
    assert ak.to_list(listoffsetarray) == [[0, 9, 3], [], [2, 2]]

    assert ak._ext._slice_tostring(listoffsetarray) == "[jagged([0, 3, 3, 5], array([0, 9, 3, 2, 2]))]"

    starts = ak.layout.Index64(np.array([1, 99, 5], dtype=np.int64))
    stops = ak.layout.Index64(np.array([4, 99, 7], dtype=np.int64))
    listarray = ak.layout.ListArray64(starts, stops, content)
    assert ak.to_list(listarray) == [[0, 9, 3], [], [2, 5]]

    assert ak._ext._slice_tostring(listarray) == "[jagged([0, 3, 3, 5], array([0, 9, 3, 2, 5]))]"

def test_missing():
    array = ak.Array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], check_valid=True)
    assert ak.to_list(array[ak.Array([3, 6, None, None, -2, 6], check_valid=True)]) == [3.3, 6.6, None, None, 8.8, 6.6]

    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 999]))
    regulararray = ak.layout.RegularArray(content, 4)
    assert ak.to_list(regulararray) == [[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7], [8.8, 9.9, 10.0, 11.1]]
    assert ak.to_list(regulararray[ak.Array([2, 1, 1, None, -1], check_valid=True)]) == [[8.8, 9.9, 10.0, 11.1], [4.4, 5.5, 6.6, 7.7], [4.4, 5.5, 6.6, 7.7], None, [8.8, 9.9, 10.0, 11.1]]
    assert ak.to_list(regulararray[:, ak.Array([2, 1, 1, None, -1], check_valid=True)]) == [[2.2, 1.1, 1.1, None, 3.3], [6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]
    assert ak.to_list(regulararray[1:, ak.Array([2, 1, 1, None, -1], check_valid=True)]) == [[6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]

    assert ak.to_list(regulararray[np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False])]) == [[8.8, 9.9, 10.0, 11.1], [4.4, 5.5, 6.6, 7.7], [4.4, 5.5, 6.6, 7.7], None, [8.8, 9.9, 10.0, 11.1]]
    assert ak.to_list(regulararray[:, np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False])]) == [[2.2, 1.1, 1.1, None, 3.3], [6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]
    assert ak.to_list(regulararray[1:, np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False])]) == [[6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]

    content = ak.layout.NumpyArray(np.array([[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7], [8.8, 9.9, 10.0, 11.1]]))
    assert ak.to_list(content[ak.Array([2, 1, 1, None, -1], check_valid=True)]) == [[8.8, 9.9, 10.0, 11.1], [4.4, 5.5, 6.6, 7.7], [4.4, 5.5, 6.6, 7.7], None, [8.8, 9.9, 10.0, 11.1]]
    assert ak.to_list(content[:, ak.Array([2, 1, 1, None, -1], check_valid=True)]) == [[2.2, 1.1, 1.1, None, 3.3], [6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]
    assert ak.to_list(content[1:, ak.Array([2, 1, 1, None, -1], check_valid=True)]) == [[6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]

    assert ak.to_list(content[np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False])]) == [[8.8, 9.9, 10.0, 11.1], [4.4, 5.5, 6.6, 7.7], [4.4, 5.5, 6.6, 7.7], None, [8.8, 9.9, 10.0, 11.1]]
    assert ak.to_list(content[:, np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False])]) == [[2.2, 1.1, 1.1, None, 3.3], [6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]
    assert ak.to_list(content[1:, np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False])]) == [[6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]

    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 999]))
    offsets = ak.layout.Index64(np.array([0, 4, 8, 12], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)
    assert ak.to_list(listoffsetarray) == [[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7], [8.8, 9.9, 10.0, 11.1]]
    assert ak.to_list(listoffsetarray[:, ak.Array([2, 1, 1, None, -1], check_valid=True)]) == [[2.2, 1.1, 1.1, None, 3.3], [6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]
    assert ak.to_list(listoffsetarray[1:, ak.Array([2, 1, 1, None, -1], check_valid=True)]) == [[6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]

    assert ak.to_list(listoffsetarray[:, np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False])]) == [[2.2, 1.1, 1.1, None, 3.3], [6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]
    assert ak.to_list(listoffsetarray[1:, np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False])]) == [[6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]

def test_bool_missing():
    data = [1.1, 2.2, 3.3, 4.4, 5.5]
    array = ak.layout.NumpyArray(np.array(data))

    assert ak._ext._slice_tostring(ak.Array([True, False, None, True, False], check_valid=True)) == "[missing([0, -1, 1], array([0, 3]))]"
    assert ak._ext._slice_tostring(ak.Array([None, None, None], check_valid=True)) == "[missing([-1, -1, -1], array([]))]"

    for x1 in [True, False, None]:
        for x2 in [True, False, None]:
            for x3 in [True, False, None]:
                for x4 in [True, False, None]:
                    for x5 in [True, False, None]:
                        mask = [x1, x2, x3, x4, x5]
                        expected = [m if m is None else x for x, m in zip(data, mask) if m is not False]
                        assert ak.to_list(array[ak.Array(mask, check_valid=True)]) == expected

def test_bool_missing2():
    array = ak.Array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], check_valid=True)
    assert ak.to_list(array[ak.Array([3, 6, None, None, -2, 6], check_valid=True)]) == [3.3, 6.6, None, None, 8.8, 6.6]

    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 999]))
    regulararray = ak.layout.RegularArray(content, 4)
    assert ak.to_list(regulararray) == [[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7], [8.8, 9.9, 10.0, 11.1]]
    assert ak.to_list(regulararray[:, ak.Array([True, None, False, True], check_valid=True)]) == [[0.0, None, 3.3], [4.4, None, 7.7], [8.8, None, 11.1]]
    assert ak.to_list(regulararray[1:, ak.Array([True, None, False, True], check_valid=True)]) == [[4.4, None, 7.7], [8.8, None, 11.1]]

    content = ak.layout.NumpyArray(np.array([[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7], [8.8, 9.9, 10.0, 11.1]]))
    assert ak.to_list(content[:, ak.Array([True, None, False, True], check_valid=True)]) == [[0.0, None, 3.3], [4.4, None, 7.7], [8.8, None, 11.1]]
    assert ak.to_list(content[1:, ak.Array([True, None, False, True], check_valid=True)]) == [[4.4, None, 7.7], [8.8, None, 11.1]]

    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 999]))
    offsets = ak.layout.Index64(np.array([0, 4, 8, 12], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)
    assert ak.to_list(listoffsetarray[:, ak.Array([True, None, False, True], check_valid=True)]) == [[0.0, None, 3.3], [4.4, None, 7.7], [8.8, None, 11.1]]
    assert ak.to_list(listoffsetarray[1:, ak.Array([True, None, False, True], check_valid=True)]) == [[4.4, None, 7.7], [8.8, None, 11.1]]

def test_records_missing():
    array = ak.Array([{"x": 0, "y": 0.0}, {"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}, {"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}, {"x": 6, "y": 6.6}, {"x": 7, "y": 7.7}, {"x": 8, "y": 8.8}, {"x": 9, "y": 9.9}], check_valid=True)
    assert ak.to_list(array[ak.Array([3, 1, None, 1, 7], check_valid=True)]) == [{"x": 3, "y": 3.3}, {"x": 1, "y": 1.1}, None, {"x": 1, "y": 1.1}, {"x": 7, "y": 7.7}]

    array = ak.Array([[{"x": 0, "y": 0.0}, {"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}, {"x": 6, "y": 6.6}, {"x": 7, "y": 7.7}, {"x": 8, "y": 8.8}, {"x": 9, "y": 9.9}]], check_valid=True)
    assert ak.to_list(array[:, ak.Array([1, None, 2, -1], check_valid=True)]) == [[{"x": 1, "y": 1.1}, None, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], [{"x": 5, "y": 5.5}, None, {"x": 6, "y": 6.6}, {"x": 9, "y": 9.9}]]

    array = ak.Array([{"x": [0, 1, 2, 3], "y": [0.0, 1.1, 2.2, 3.3]}, {"x": [4, 5, 6, 7], "y": [4.4, 5.5, 6.6, 7.7]}, {"x": [8, 9, 10, 11], "y": [8.8, 9.9, 10.0, 11.1]}], check_valid=True)
    assert ak.to_list(array[:, ak.Array([1, None, 2, -1], check_valid=True)]) == [{"x": [1, None, 2, 3], "y": [1.1, None, 2.2, 3.3]}, {"x": [5, None, 6, 7], "y": [5.5, None, 6.6, 7.7]}, {"x": [9, None, 10, 11], "y": [9.9, None, 10.0, 11.1]}]
    assert ak.to_list(array[1:, ak.Array([1, None, 2, -1], check_valid=True)]) == [{"x": [5, None, 6, 7], "y": [5.5, None, 6.6, 7.7]}, {"x": [9, None, 10, 11], "y": [9.9, None, 10.0, 11.1]}]

def test_jagged():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], check_valid=True)
    assert ak.to_list(array[ak.Array([[0, -1], [], [-1, 0], [-1], [1, 1, -2, 0]], check_valid=True)]) == [[1.1, 3.3], [], [5.5, 4.4], [6.6], [8.8, 8.8, 8.8, 7.7]]

def test_double_jagged():
    array = ak.Array([[[0, 1, 2, 3], [4, 5]], [[6, 7, 8], [9, 10, 11, 12, 13]]], check_valid=True)
    assert ak.to_list(array[ak.Array([[[2, 1, 0], [-1]], [[-1, -2, -3], [2, 1, 1, 3]]], check_valid=True)]) == [[[2, 1, 0], [5]], [[8, 7, 6], [11, 10, 10, 12]]]

    content = ak.from_iter([[0, 1, 2, 3], [4, 5], [6, 7, 8], [9, 10, 11, 12, 13]], highlevel=False)
    regulararray = ak.layout.RegularArray(content, 2)

    assert ak.to_list(regulararray[:, ak.Array([[2, 1, 0], [-1]], check_valid=True)]) == [[[2, 1, 0], [5]], [[8, 7, 6], [13]]]
    assert ak.to_list(regulararray[1:, ak.Array([[2, 1, 0], [-1]], check_valid=True)]) == [[[8, 7, 6], [13]]]

    offsets = ak.layout.Index64(np.array([0, 2, 4], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)
    assert ak.to_list(listoffsetarray[:, ak.Array([[2, 1, 0], [-1]], check_valid=True)]) == [[[2, 1, 0], [5]], [[8, 7, 6], [13]]]
    assert ak.to_list(listoffsetarray[1:, ak.Array([[2, 1, 0], [-1]], check_valid=True)]) == [[[8, 7, 6], [13]]]

def test_masked_jagged():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], check_valid=True)
    assert ak.to_list(array[ak.Array([[-1, -2], None, [], None, [-2, 0]], check_valid=True)]) == [[3.3, 2.2], None, [], None, [8.8, 7.7]]

def test_jagged_masked():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], check_valid=True)
    assert ak.to_list(array[ak.Array([[-1, None], [], [None, 0], [None], [1]], check_valid=True)]) == [[3.3, None], [], [None, 4.4], [None], [8.8]]

def test_regular_regular():
    content = ak.layout.NumpyArray(np.arange(2*3*5))
    regulararray1 = ak.layout.RegularArray(content, 5)
    regulararray2 = ak.layout.RegularArray(regulararray1, 3)

    assert ak.to_list(regulararray2[ak.Array([[[2], [1, -2], [-1, 2, 0]], [[-3], [-4, 3], [-5, -3, 4]]], check_valid=True)]) == [[[2], [6, 8], [14, 12, 10]], [[17], [21, 23], [25, 27, 29]]]

    assert ak.to_list(regulararray2[ak.Array([[[2], [1, -2], [-1, None, 0]], [[-3], [-4, 3], [-5, None, 4]]], check_valid=True)]) == [[[2], [6, 8], [14, None, 10]], [[17], [21, 23], [25, None, 29]]]

def test_masked_of_jagged_of_whatever():
    content = ak.layout.NumpyArray(np.arange(2*3*5))
    regulararray1 = ak.layout.RegularArray(content, 5)
    regulararray2 = ak.layout.RegularArray(regulararray1, 3)

    assert ak.to_list(regulararray2[ak.Array([[[2], None, [-1, 2, 0]], [[-3], None, [-5, -3, 4]]], check_valid=True)]) == [[[2], None, [14, 12, 10]], [[17], None, [25, 27, 29]]]

    assert ak.to_list(regulararray2[ak.Array([[[2], None, [-1, None, 0]], [[-3], None, [-5, None, 4]]], check_valid=True)]) == [[[2], None, [14, None, 10]], [[17], None, [25, None, 29]]]

def test_emptyarray():
    content = ak.layout.EmptyArray()
    offsets = ak.layout.Index64(np.array([0, 0, 0, 0, 0], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)
    assert ak.to_list(listoffsetarray) == [[], [], [], []]

    assert ak.to_list(listoffsetarray[ak.Array([[], [], [], []], check_valid=True)]) == [[], [], [], []]
    assert ak.to_list(listoffsetarray[ak.Array([[], [None], [], []], check_valid=True)]) == [[], [None], [], []]
    assert ak.to_list(listoffsetarray[ak.Array([[], [], None, []], check_valid=True)]) == [[], [], None, []]
    assert ak.to_list(listoffsetarray[ak.Array([[], [None], None, []], check_valid=True)]) == [[], [None], None, []]

    with pytest.raises(ValueError):
        listoffsetarray[ak.Array([[], [0], [], []], check_valid=True)]

def test_numpyarray():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)
    with pytest.raises(ValueError):
        array[ak.Array([[[], [], []], [], [[], []]], check_valid=True)]

def test_record():
    array = ak.Array([{"x": [0, 1, 2], "y": [0.0, 1.1, 2.2, 3.3]}, {"x": [3, 4, 5, 6], "y": [4.4, 5.5]}, {"x": [7, 8], "y": [6.6, 7.7, 8.8, 9.9]}], check_valid=True)
    assert ak.to_list(array[ak.Array([[-1, 1], [0, 0, 1], [-1, -2]], check_valid=True)]) == [{"x": [2, 1], "y": [3.3, 1.1]}, {"x": [3, 3, 4], "y": [4.4, 4.4, 5.5]}, {"x": [8, 7], "y": [9.9, 8.8]}]
    assert ak.to_list(array[ak.Array([[-1, 1], [0, 0, None, 1], [-1, -2]], check_valid=True)]) == [{"x": [2, 1], "y": [3.3, 1.1]}, {"x": [3, 3, None, 4], "y": [4.4, 4.4, None, 5.5]}, {"x": [8, 7], "y": [9.9, 8.8]}]
    assert ak.to_list(array[ak.Array([[-1, 1], None, [-1, -2]], check_valid=True)]) == [{"x": [2, 1], "y": [3.3, 1.1]}, None, {"x": [8, 7], "y": [9.9, 8.8]}]

def test_indexedarray():
    array = ak.from_iter([[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False)
    index = ak.layout.Index64(np.array([3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, array)
    assert ak.to_list(indexedarray) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], [0.0, 1.1, 2.2]]

    assert ak.to_list(indexedarray[ak.Array([[0, -1], [0], [], [1, 1]], check_valid=True)]) == [[6.6, 9.9], [5.5], [], [1.1, 1.1]]
    assert ak.to_list(indexedarray[ak.Array([[0, -1], [0], [None], [1, None, 1]], check_valid=True)]) == [[6.6, 9.9], [5.5], [None], [1.1, None, 1.1]]
    assert ak.to_list(indexedarray[ak.Array([[0, -1], [0], None, [1, 1]], check_valid=True)]) == [[6.6, 9.9], [5.5], None, [1.1, 1.1]]
    assert ak.to_list(indexedarray[ak.Array([[0, -1], [0], None, [None]], check_valid=True)]) == [[6.6, 9.9], [5.5], None, [None]]

    index = ak.layout.Index64(np.array([3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedOptionArray64(index, array)
    assert ak.to_list(indexedarray) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], [0.0, 1.1, 2.2]]

    assert ak.to_list(indexedarray[ak.Array([[0, -1], [0], [], [1, 1]], check_valid=True)]) == [[6.6, 9.9], [5.5], [], [1.1, 1.1]]
    assert ak.to_list(indexedarray[ak.Array([[0, -1], [0], [None], [1, None, 1]], check_valid=True)]) == [[6.6, 9.9], [5.5], [None], [1.1, None, 1.1]]
    assert ak.to_list(indexedarray[ak.Array([[0, -1], [0], None, []], check_valid=True)]) == [[6.6, 9.9], [5.5], None, []]
    assert ak.to_list(indexedarray[ak.Array([[0, -1], [0], None, [1, None, 1]], check_valid=True)]) == [[6.6, 9.9], [5.5], None, [1.1, None, 1.1]]

def test_indexedarray2():
    array = ak.from_iter([[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False)
    index = ak.layout.Index64(np.array([3, 2, -1, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedOptionArray64(index, array)
    assert ak.to_list(indexedarray) == [[6.6, 7.7, 8.8, 9.9], [5.5], None, [0.0, 1.1, 2.2]]
    assert ak.to_list(indexedarray[ak.Array([[0, -1], [0], None, [1, 1]])]) == [[6.6, 9.9], [5.5], None, [1.1, 1.1]]

def test_indexedarray2b():
    array = ak.from_iter([[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False)
    index = ak.layout.Index64(np.array([0, -1, 2, 3], dtype=np.int64))
    indexedarray = ak.layout.IndexedOptionArray64(index, array)
    assert ak.to_list(indexedarray) == [[0.0, 1.1, 2.2], None, [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert ak.to_list(indexedarray[ak.Array([[1, 1], None, [0], [0, -1]])]) == [[1.1, 1.1], None, [5.5], [6.6, 9.9]]

def test_bytemaskedarray2b():
    array = ak.from_iter([[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False)
    mask = ak.layout.Index8(np.array([0, 1, 0, 0], dtype=np.int8))
    maskedarray = ak.layout.ByteMaskedArray(mask, array, valid_when=False)
    assert ak.to_list(maskedarray) == [[0.0, 1.1, 2.2], None, [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert ak.to_list(maskedarray[ak.Array([[1, 1], None, [0], [0, -1]])]) == [[1.1, 1.1], None, [5.5], [6.6, 9.9]]

def test_bitmaskedarray2b():
    array = ak.from_iter([[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False)
    mask = ak.layout.IndexU8(np.array([66], dtype=np.uint8))
    maskedarray = ak.layout.BitMaskedArray(mask, array, valid_when=False, length=4, lsb_order=True)  # lsb_order is irrelevant in this example
    assert ak.to_list(maskedarray) == [[0.0, 1.1, 2.2], None, [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert ak.to_list(maskedarray[ak.Array([[1, 1], None, [0], [0, -1]])]) == [[1.1, 1.1], None, [5.5], [6.6, 9.9]]

def test_indexedarray3():
    array = ak.Array([0.0, 1.1, 2.2, None, 4.4, None, None, 7.7])
    assert ak.to_list(array[ak.Array([4, 3, 2])]) == [4.4, None, 2.2]
    assert ak.to_list(array[ak.Array([4, 3, 2, None, 1])]) == [4.4, None, 2.2, None, 1.1]

    array = ak.Array([[0.0, 1.1, None, 2.2], [3.3, None, 4.4], [5.5]])
    assert ak.to_list(array[ak.Array([[3, 2, 2, 1], [1, 2], []])]) == [[2.2, None, None, 1.1], [None, 4.4], []]

    array = ak.Array([[0.0, 1.1, 2.2], [3.3, 4.4], None, [5.5]])
    assert ak.to_list(array[ak.Array([3, 2, 1])]) == [[5.5], None, [3.3, 4.4]]
    assert ak.to_list(array[ak.Array([3, 2, 1, None, 0])]) == [[5.5], None, [3.3, 4.4], None, [0.0, 1.1, 2.2]]

    assert (ak.to_list(array[ak.Array([[2, 1, 1, 0], [1], None, [0]])])) == [[2.2, 1.1, 1.1, 0.0], [4.4], None, [5.5]]

    assert ak.to_list(array[ak.Array([[2, 1, 1, 0], None, [1], [0]])]) == [[2.2, 1.1, 1.1, 0], None, None, [5.5]]
    with pytest.raises(ValueError):
        array[ak.Array([[2, 1, 1, 0], None, [1], [0], None])]

def test_sequential():
    array = ak.Array(np.arange(2*3*5).reshape(2, 3, 5).tolist(), check_valid=True)
    assert ak.to_list(array[ak.Array([[2, 1, 0], [2, 1, 0]], check_valid=True)]) == [[[10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]], [[25, 26, 27, 28, 29], [20, 21, 22, 23, 24], [15, 16, 17, 18, 19]]]
    assert ak.to_list(array[ak.Array([[2, 1, 0], [2, 1, 0]], check_valid=True), :2]) == [[[10, 11], [5, 6], [0, 1]], [[25, 26], [20, 21], [15, 16]]]

def test_union():
    one = ak.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    two = ak.from_iter([[6.6], [7.7, 8.8], [], [9.9, 10.0, 11.1, 12.2]], highlevel=False)
    tags = ak.layout.Index8(np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int8))
    index = ak.layout.Index64(np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int64))
    unionarray = ak.layout.UnionArray8_64(tags, index, [one, two])
    assert ak.to_list(unionarray) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8], [], [9.9, 10.0, 11.1, 12.2]]

    assert ak.to_list(unionarray[ak.Array([[0, -1], [], [1, 1], [], [-1], [], [1, -2, -1]], check_valid=True)]) == [[1.1, 3.3], [], [5.5, 5.5], [], [8.8], [], [10.0, 11.1, 12.2]]

def test_python_to_jaggedslice():
    assert ak._ext._slice_tostring([[1, 2, 3], [], [4, 5]]) == "[jagged([0, 3, 3, 5], array([1, 2, 3, 4, 5]))]"
    assert ak._ext._slice_tostring([[1, 2], [3, 4], [5, 6]]) == "[array([[1, 2], [3, 4], [5, 6]])]"

def test_jagged_mask():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], check_valid=True)
    assert ak.to_list(array[[[True, True, True], [], [True, True], [True], [True, True, True]]]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert ak.to_list(array[[[False, True, True], [], [True, True], [True], [True, True, True]]]) == [[2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert ak.to_list(array[[[True, False, True], [], [True, True], [True], [True, True, True]]]) == [[1.1, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert ak.to_list(array[[[True, True, True], [], [False, True], [True], [True, True, True]]]) == [[1.1, 2.2, 3.3], [], [5.5], [6.6], [7.7, 8.8, 9.9]]
    assert ak.to_list(array[[[True, True, True], [], [False, False], [True], [True, True, True]]]) == [[1.1, 2.2, 3.3], [], [], [6.6], [7.7, 8.8, 9.9]]

def test_jagged_missing_mask():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)
    assert ak.to_list(array[[[True, True, True], [], [True, True]]]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert ak.to_list(array[[[True, False, True], [], [True, True]]]) == [[1.1, 3.3], [], [4.4, 5.5]]
    assert ak.to_list(array[[[True, None, True], [], [True, True]]]) == [[1.1, None, 3.3], [], [4.4, 5.5]]
    assert ak.to_list(array[[[True, None, False], [], [True, True]]]) == [[1.1, None], [], [4.4, 5.5]]
    assert ak.to_list(array[[[False, None, True], [], [True, True]]]) == [[None, 3.3], [], [4.4, 5.5]]
    assert ak.to_list(array[[[False, None, False], [], [True, True]]]) == [[None], [], [4.4, 5.5]]
    assert ak.to_list(array[[[True, True, False], [], [False, True]]]) == [[1.1, 2.2], [], [5.5]]
    assert ak.to_list(array[[[True, True, None], [], [False, True]]]) == [[1.1, 2.2, None], [], [5.5]]
    assert ak.to_list(array[[[True, True, False], [None], [False, True]]]) == [[1.1, 2.2], [None], [5.5]]
    assert ak.to_list(array[[[True, True, False], [], [None, True]]]) == [[1.1, 2.2], [], [None, 5.5]]
    assert ak.to_list(array[[[True, True, False], [], [True, None]]]) == [[1.1, 2.2], [], [4.4, None]]
    assert ak.to_list(array[[[True, True, False], [], [False, None]]]) == [[1.1, 2.2], [], [None]]
