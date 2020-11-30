# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import pytest
import numpy as np
import awkward1 as ak

def test_rpad_and_clip_empty_array():
    empty = ak.layout.EmptyArray()

    assert ak.to_list(empty) == []
    assert ak.to_list(empty.rpad(5, 0)) == [None, None, None, None, None]
    assert ak.to_list(empty.rpad_and_clip(5, 0)) == [None, None, None, None, None]

def test_rpad_and_clip_numpy_array():
    array = ak.layout.NumpyArray(np.arange(2*3*5, dtype=np.int64).reshape(2, 3, 5))
    assert ak.to_list(array) == [[[ 0,  1,  2,  3,  4], [ 5,  6,  7,  8,  9], [10, 11, 12, 13, 14]],
                                      [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]

    assert ak.to_list(array.rpad_and_clip(5, 0)) == [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
                                                [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
                                                None, None, None]

    assert ak.to_list(array.rpad_and_clip(5, 1)) == [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], None, None],
                                                [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29], None, None]]

    assert ak.to_list(array.rpad_and_clip(7, 2)) == [[[0, 1, 2, 3, 4, None, None], [5, 6, 7, 8, 9, None, None], [10, 11, 12, 13, 14, None, None]],
                                                [[15, 16, 17, 18, 19, None, None], [20, 21, 22, 23, 24, None, None], [25, 26, 27, 28, 29, None, None]]]

    assert ak.to_list(array.rpad_and_clip(2, 2)) == [[[0, 1], [5, 6], [10, 11]], [[15, 16], [20, 21], [25, 26]]]

def test_rpad_numpy_array():
    array = ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    assert ak.to_list(array.rpad(10, 0)) == [1.1, 2.2, 3.3, 4.4, 5.5, None, None, None, None, None]

    array = ak.layout.NumpyArray(np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]))
    assert ak.to_list(array.rpad(5, 0)) == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], None, None, None]
    assert ak.to_list(array.rpad(5, 1)) == [[1.1, 2.2, 3.3, None, None], [4.4, 5.5, 6.6, None, None]]

    array = ak.layout.NumpyArray(np.arange(2*3*5, dtype=np.int64).reshape(2, 3, 5))
    assert ak.to_list(array) == [[[ 0,  1,  2,  3,  4], [ 5,  6,  7,  8,  9], [10, 11, 12, 13, 14]],
                                      [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]

    assert ak.to_list(array.rpad(1, 0)) == [[[ 0,  1,  2,  3,  4], [ 5,  6,  7,  8,  9], [10, 11, 12, 13, 14]],
                                                [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]
    assert ak.to_list(array.rpad(2, 0)) == [[[ 0,  1,  2,  3,  4], [ 5,  6,  7,  8,  9], [10, 11, 12, 13, 14]],
                                                [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]
    assert ak.to_list(array.rpad(3, 0)) == [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
                                                [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
                                                None]
    assert ak.to_list(array.rpad(4, 0)) == [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
                                                [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
                                                None, None]
    assert ak.to_list(array.rpad(5, 0)) == [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
                                                [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
                                                None, None, None]

    assert ak.to_list(array.rpad(2, 1)) == [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
                                                [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]
    assert ak.to_list(array.rpad(3, 1)) == [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
                                                [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]
    assert ak.to_list(array.rpad(4, 1)) == [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], None],
                                                [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29], None]]
    assert ak.to_list(array.rpad(5, 1)) == [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], None, None],
                                                [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29], None, None]]

    assert ak.to_list(array.rpad(3, 2)) == [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
                                                [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]

    assert ak.to_list(array.rpad(7, 2)) == [[[0, 1, 2, 3, 4, None, None], [5, 6, 7, 8, 9, None, None], [10, 11, 12, 13, 14, None, None]],
                                                [[15, 16, 17, 18, 19, None, None], [20, 21, 22, 23, 24, None, None], [25, 26, 27, 28, 29, None, None]]]

    assert ak.to_list(array.rpad(2, 2)) == [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
                                                 [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]

def test_rpad_and_clip_regular_array():
    content = ak.layout.NumpyArray(np.array([2.1, 8.4, 7.4, 1.6, 2.2, 3.4, 6.2, 5.4, 1.5, 3.9, 3.8, 3.0, 8.5, 6.9, 4.3, 3.6, 6.7, 1.8, 3.2]))
    index = ak.layout.Index64(np.array([13, 9, 13, 4, 8, 3, 15, -1, 16, 2, 8], dtype=np.int64))
    indexedarray = ak.layout.IndexedOptionArray64(index, content)
    array = ak.layout.RegularArray(indexedarray, 3)

    assert ak.to_list(array.rpad_and_clip(5, 0)) == [[6.9, 3.9, 6.9], [2.2, 1.5, 1.6], [3.6, None, 6.7], None, None]
    assert ak.to_list(array.rpad_and_clip(4, 0)) == [[6.9, 3.9, 6.9], [2.2, 1.5, 1.6], [3.6, None, 6.7], None]
    assert ak.to_list(array.rpad_and_clip(3, 0)) == [[6.9, 3.9, 6.9], [2.2, 1.5, 1.6], [3.6, None, 6.7]]
    assert ak.to_list(array.rpad_and_clip(2, 0)) == [[6.9, 3.9, 6.9], [2.2, 1.5, 1.6]]
    assert ak.to_list(array.rpad_and_clip(1, 0)) == [[6.9, 3.9, 6.9]]
    assert ak.to_list(array.rpad_and_clip(5, 1)) == [[6.9, 3.9, 6.9, None, None], [2.2, 1.5, 1.6, None, None], [3.6, None, 6.7, None, None]]
    assert ak.to_list(array.rpad_and_clip(4, 1)) == [[6.9, 3.9, 6.9, None], [2.2, 1.5, 1.6, None], [3.6, None, 6.7, None]]
    assert ak.to_list(array.rpad_and_clip(3, 1)) == [[6.9, 3.9, 6.9], [2.2, 1.5, 1.6], [3.6, None, 6.7]]
    assert ak.to_list(array.rpad_and_clip(2, 1)) == [[6.9, 3.9], [2.2, 1.5], [3.6, None]]
    assert ak.to_list(array.rpad_and_clip(1, 1)) == [[6.9], [2.2], [3.6]]

    array = ak.layout.NumpyArray(np.arange(2*3*5).reshape(2, 3, 5))
    assert ak.to_list(array) == [[[ 0,  1,  2,  3,  4],
                                       [ 5,  6,  7,  8,  9],
                                       [10, 11, 12, 13, 14]],
                                      [[15, 16, 17, 18, 19],
                                       [20, 21, 22, 23, 24],
                                       [25, 26, 27, 28, 29]]]

    assert ak.to_list(array.rpad_and_clip(7, 2)) == [[[ 0,  1,  2,  3,  4, None, None],
                                                           [ 5,  6,  7,  8,  9, None, None],
                                                           [10, 11, 12, 13, 14, None, None]],
                                                          [[15, 16, 17, 18, 19, None, None],
                                                           [20, 21, 22, 23, 24, None, None],
                                                           [25, 26, 27, 28, 29, None, None]]]

    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)
    regulararray = ak.layout.RegularArray(listoffsetarray, 2)

    assert ak.to_list(regulararray.rpad_and_clip(1, 0)) == [[[0.0, 1.1, 2.2], []]]
    assert ak.to_list(regulararray.rpad_and_clip(2, 0)) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]]]
    assert ak.to_list(regulararray.rpad_and_clip(3, 0)) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]
    assert ak.to_list(regulararray.rpad_and_clip(4, 0)) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []], None]
    assert ak.to_list(regulararray.rpad_and_clip(5, 0)) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []], None, None]

    assert ak.to_list(regulararray.rpad_and_clip(1, 1)) == [[[0.0, 1.1, 2.2]], [[3.3, 4.4]], [[6.6, 7.7, 8.8, 9.9]]]
    assert ak.to_list(regulararray.rpad_and_clip(2, 1)) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]
    assert ak.to_list(regulararray.rpad_and_clip(3, 1)) == [[[0.0, 1.1, 2.2], [], None], [[3.3, 4.4], [5.5], None], [[6.6, 7.7, 8.8, 9.9], [], None]]
    assert ak.to_list(regulararray.rpad_and_clip(7, 1)) == [[[0.0, 1.1, 2.2], [], None, None, None, None, None], [[3.3, 4.4], [5.5], None, None, None, None, None], [[6.6, 7.7, 8.8, 9.9], [], None, None, None, None, None]]

    assert ak.to_list(regulararray.rpad_and_clip(1, 2)) == [[[0.0], [None]], [[3.3], [5.5]], [[6.6], [None]]]
    assert ak.to_list(regulararray.rpad_and_clip(2, 2)) == [[[0.0, 1.1], [None, None]], [[3.3, 4.4], [5.5, None]], [[6.6, 7.7], [None, None]]]
    assert ak.to_list(regulararray.rpad_and_clip(3, 2)) == [[[0.0, 1.1, 2.2], [None, None, None]], [[3.3, 4.4, None], [5.5, None, None]], [[6.6, 7.7, 8.8], [None, None, None]]]
    assert ak.to_list(regulararray.rpad_and_clip(4, 2)) == [[[0.0, 1.1, 2.2, None], [None, None, None, None]], [[3.3, 4.4, None, None], [5.5, None, None, None]], [[6.6, 7.7, 8.8, 9.9], [None, None, None, None]]]
    assert ak.to_list(regulararray.rpad_and_clip(5, 2)) == [[[0.0, 1.1, 2.2, None, None], [None, None, None, None, None]], [[3.3, 4.4, None, None, None], [5.5, None, None, None, None]], [[6.6, 7.7, 8.8, 9.9, None], [None, None, None, None, None]]]

def test_rpad_regular_array():
    content = ak.layout.NumpyArray(np.array([2.1, 8.4, 7.4, 1.6, 2.2, 3.4, 6.2, 5.4, 1.5, 3.9, 3.8, 3.0, 8.5, 6.9, 4.3, 3.6, 6.7, 1.8, 3.2]))
    index = ak.layout.Index64(np.array([13, 9, 13, 4, 8, 3, 15, -1, 16, 2, 8], dtype=np.int64))
    indexedarray = ak.layout.IndexedOptionArray64(index, content)
    array = ak.layout.RegularArray(indexedarray, 3)

    assert ak.to_list(array.rpad(5, 0)) == [[6.9, 3.9, 6.9], [2.2, 1.5, 1.6], [3.6, None, 6.7], None, None]
    assert ak.to_list(array.rpad(4, 0)) == [[6.9, 3.9, 6.9], [2.2, 1.5, 1.6], [3.6, None, 6.7], None]
    assert ak.to_list(array.rpad(3, 0)) == [[6.9, 3.9, 6.9], [2.2, 1.5, 1.6], [3.6, None, 6.7]]
    assert ak.to_list(array.rpad(1, 0)) == [[6.9, 3.9, 6.9], [2.2, 1.5, 1.6], [3.6, None, 6.7]]
    assert ak.to_list(array.rpad(5, 1)) == [[6.9, 3.9, 6.9, None, None], [2.2, 1.5, 1.6, None, None], [3.6, None, 6.7, None, None]]
    assert ak.to_list(array.rpad(4, 1)) == [[6.9, 3.9, 6.9, None], [2.2, 1.5, 1.6, None], [3.6, None, 6.7, None]]
    assert ak.to_list(array.rpad(3, 1)) == [[6.9, 3.9, 6.9], [2.2, 1.5, 1.6], [3.6, None, 6.7]]
    assert ak.to_list(array.rpad(1, 1)) == [[6.9, 3.9, 6.9], [2.2, 1.5, 1.6], [3.6, None, 6.7]]

    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)
    regulararray = ak.layout.RegularArray(listoffsetarray, 2)

    assert ak.to_list(regulararray.rpad(1, 0)) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]
    assert ak.to_list(regulararray.rpad(3, 0)) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]
    assert ak.to_list(regulararray.rpad(4, 0)) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []], None]
    assert ak.to_list(regulararray.rpad(7, 0)) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []], None, None, None, None]

    assert ak.to_list(regulararray.rpad(1, 1)) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]
    assert ak.to_list(regulararray.rpad(2, 1)) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]
    assert ak.to_list(regulararray.rpad(3, 1)) == [[[0.0, 1.1, 2.2], [], None], [[3.3, 4.4], [5.5], None], [[6.6, 7.7, 8.8, 9.9], [], None]]
    assert ak.to_list(regulararray.rpad(5, 1)) == [[[0.0, 1.1, 2.2], [], None, None, None], [[3.3, 4.4], [5.5], None, None, None], [[6.6, 7.7, 8.8, 9.9], [], None, None, None]]
    assert ak.to_list(regulararray.rpad(7, 1)) == [[[0.0, 1.1, 2.2], [], None, None, None, None, None], [[3.3, 4.4], [5.5], None, None, None, None, None], [[6.6, 7.7, 8.8, 9.9], [], None, None, None, None, None]]

    assert ak.to_list(regulararray.rpad(1, 2)) == [[[0.0, 1.1, 2.2], [None]], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], [None]]]
    assert ak.to_list(regulararray.rpad(2, 2)) == [[[0.0, 1.1, 2.2], [None, None]], [[3.3, 4.4], [5.5, None]], [[6.6, 7.7, 8.8, 9.9], [None, None]]]
    assert ak.to_list(regulararray.rpad(3, 2)) == [[[0.0, 1.1, 2.2], [None, None, None]], [[3.3, 4.4, None], [5.5, None, None]], [[6.6, 7.7, 8.8, 9.9], [None, None, None]]]
    assert ak.to_list(regulararray.rpad(4, 2)) == [[[0.0, 1.1, 2.2, None], [None, None, None, None]], [[3.3, 4.4, None, None], [5.5, None, None, None]], [[6.6, 7.7, 8.8, 9.9], [None, None, None, None]]]

def test_rpad_and_clip_listoffset_array():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)
    assert ak.to_list(listoffsetarray) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9], []]

    assert ak.to_list(listoffsetarray.rpad_and_clip(3,0)) == [[0.0, 1.1, 2.2], [], [3.3, 4.4]]
    assert str("option[") + str(ak.type(listoffsetarray)) + str("]") == str(ak.type(listoffsetarray.rpad_and_clip(3,0)))

    assert ak.to_list(listoffsetarray.rpad_and_clip(7,0)) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9], [], None]
    assert str("option[") + str(ak.type(listoffsetarray)) + str("]") == str(ak.type(listoffsetarray.rpad_and_clip(7,0)))

    assert ak.to_list(listoffsetarray.rpad_and_clip(5,1)) == [[0.0, 1.1, 2.2, None, None], [None, None, None, None, None], [3.3, 4.4, None, None, None], [5.5, None, None, None, None], [6.6, 7.7, 8.8, 9.9, None], [None, None, None, None, None]]

    assert str(ak.type(listoffsetarray.rpad(5, 1))) == "var * ?float64"
    assert str(ak.type(listoffsetarray.rpad_and_clip(5, 1))) == "5 * ?float64"

    assert ak.to_list(listoffsetarray.rpad_and_clip(1,1)) == [[0.0], [None], [3.3], [5.5], [6.6], [None]]

    content = ak.layout.NumpyArray(np.array([1.5, 3.3]))
    index = ak.layout.Index64(np.array([0, -3, 1, -2, 1, 0, 0, -3, -13, 0, 1, 1, 0, 1, 1, 1, 1, -10, 0, -1, 0, 0, 0, 1, -1, 1, 1]))
    indexedarray = ak.layout.IndexedOptionArray64(index, content)
    offsets = ak.layout.Index64(np.array([14, 15, 15, 15, 26, 26, 26]))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, indexedarray)

    assert ak.to_list(listoffsetarray) == [[3.3], [], [], [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3], [], []]
    assert ak.to_list(listoffsetarray.rpad_and_clip(1,0)) == [[3.3]]
    assert ak.to_list(listoffsetarray.rpad_and_clip(1,1)) == [[3.3], [None], [None], [3.3], [None], [None]]

def test_rpad_listoffset_array():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)

    assert ak.to_list(listoffsetarray) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9], []]

    assert ak.to_list(listoffsetarray.rpad(3,0)) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9], []]
    assert str(ak.type(listoffsetarray)) == str(ak.type(listoffsetarray.rpad(3,0)))

    assert ak.to_list(listoffsetarray.rpad(7,0)) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9], [], None]
    assert str("option[") + str(ak.type(listoffsetarray)) + str("]") == str(ak.type(listoffsetarray.rpad(7,0)))

    assert ak.to_list(listoffsetarray.rpad(5,1)) == [[0.0, 1.1, 2.2, None, None], [None, None, None, None, None], [3.3, 4.4, None, None, None], [5.5, None, None, None, None], [6.6, 7.7, 8.8, 9.9, None], [None, None, None, None, None]]

    assert ak.to_list(listoffsetarray.rpad(1,1)) == [[0.0, 1.1, 2.2], [None], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9], [None]]

    content = ak.layout.NumpyArray(np.array([1.5, 3.3]))
    index = ak.layout.Index64(np.array([0, -3, 1, -2, 1, 0, 0, -3, -13, 0, 1, 1, 0, 1, 1, 1, 1, -10, 0, -1, 0, 0, 0, 1, -1, 1, 1]))
    indexedarray = ak.layout.IndexedOptionArray64(index, content)
    offsets = ak.layout.Index64(np.array([14, 15, 15, 15, 26, 26, 26]))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, indexedarray)

    assert ak.to_list(listoffsetarray) == [[3.3], [], [], [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3], [], []]

    assert ak.to_list(listoffsetarray.rpad(1,0)) == [[3.3], [], [], [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3], [], []]
    assert str(ak.type(listoffsetarray)) == str(ak.type(listoffsetarray.rpad(1,0)))

    assert ak.to_list(listoffsetarray.rpad(6,0)) == [[3.3], [], [], [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3], [], []]
    assert str("option[") + str(ak.type(listoffsetarray)) + str("]") == str(ak.type(listoffsetarray.rpad(6,0)))

    assert ak.to_list(listoffsetarray.rpad(7,0)) == [[3.3], [], [], [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3], [], [], None]
    assert str("option[") + str(ak.type(listoffsetarray)) + str("]") == str(ak.type(listoffsetarray.rpad(7,0)))

    assert ak.to_list(listoffsetarray.rpad(9,0)) == [[3.3], [], [], [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3], [], [], None, None, None]
    assert str("option[") + str(ak.type(listoffsetarray)) + str("]") == str(ak.type(listoffsetarray.rpad(9,0)))

    assert ak.to_list(listoffsetarray.rpad(1,1)) == [[3.3], [None], [None], [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3], [None], [None]]
    assert str(ak.type(listoffsetarray)) == str(ak.type(listoffsetarray.rpad(1,1)))

    assert ak.to_list(listoffsetarray.rpad(4,1)) == [[3.3, None, None, None], [None, None, None, None], [None, None, None, None], [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3], [None, None, None, None], [None, None, None, None]]
    assert str(ak.type(listoffsetarray)) == str(ak.type(listoffsetarray.rpad(4,1)))

def test_rpad_list_array():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    starts  = ak.layout.Index64(np.array([0, 3, 4, 5, 8]))
    stops   = ak.layout.Index64(np.array([3, 3, 6, 8, 9]))
    array   = ak.layout.ListArray64(starts, stops, content)

    assert ak.to_list(array) == [[0.0, 1.1, 2.2], [], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8]]
    assert ak.to_list(array.rpad(1,0)) == [[0.0, 1.1, 2.2], [], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8]]
    assert str(ak.type(array)) == str(ak.type(array.rpad(1,0)))

    assert ak.to_list(array.rpad(2,0)) == [[0.0, 1.1, 2.2], [], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8]]
    assert str(ak.type(array)) == str(ak.type(array.rpad(2,0)))

    assert ak.to_list(array.rpad(7,0)) == [[0.0, 1.1, 2.2], [], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8], None, None]
    assert str("option[") + str(ak.type(array)) + str("]") == str(ak.type(array.rpad(7,0)))

    assert ak.to_list(array.rpad(1,1)) == [[0.0, 1.1, 2.2], [None], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8]]

    assert ak.to_list(array.rpad(2,1)) == [[0.0, 1.1, 2.2], [None, None], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8, None]]

    assert ak.to_list(array.rpad(3,1)) == [[0.0, 1.1, 2.2], [None, None, None], [4.4, 5.5, None], [5.5, 6.6, 7.7], [8.8, None, None]]

    assert ak.to_list(array.rpad(4,1)) == [[0.0, 1.1, 2.2, None], [None, None, None, None], [4.4, 5.5, None, None], [5.5, 6.6, 7.7, None], [8.8, None, None, None]]

def test_rpad_and_clip_list_array():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    starts  = ak.layout.Index64(np.array([0, 3, 4, 5, 8]))
    stops   = ak.layout.Index64(np.array([3, 3, 6, 8, 9]))
    array   = ak.layout.ListArray64(starts, stops, content)

    assert ak.to_list(array) == [[0.0, 1.1, 2.2], [], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8]]
    assert ak.to_list(array.rpad_and_clip(1,0)) == [[0.0, 1.1, 2.2]]
    assert str("option[") + str(ak.type(array)) + str("]") == str(ak.type(array.rpad_and_clip(1,0)))

    assert ak.to_list(array.rpad_and_clip(2,0)) == [[0.0, 1.1, 2.2], []]
    assert str("option[") + str(ak.type(array)) + str("]") == str(ak.type(array.rpad_and_clip(2,0)))

    assert ak.to_list(array.rpad_and_clip(7,0)) == [[0.0, 1.1, 2.2], [], [4.4, 5.5], [5.5, 6.6, 7.7], [8.8], None, None]
    assert str("option[") + str(ak.type(array)) + str("]") == str(ak.type(array.rpad_and_clip(7,0)))

    assert ak.to_list(array.rpad_and_clip(1,1)) == [[0.0], [None], [4.4], [5.5], [8.8]]

    assert ak.to_list(array.rpad_and_clip(2,1)) == [[0.0, 1.1], [None, None], [4.4, 5.5], [5.5, 6.6], [8.8, None]]

def test_rpad_indexed_array():
    listoffsetarray = ak.from_iter([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False)
    backward = ak.from_iter([[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], [], [0.0, 1.1, 2.2]], highlevel=False)

    index = ak.layout.Index64(np.array([4, 3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, listoffsetarray)
    assert ak.to_list(indexedarray) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], [], [0.0, 1.1, 2.2]]

    assert ak.to_list(backward.rpad(4, 1)) == ak.to_list(indexedarray.rpad(4, 1))
    assert ak.to_list(indexedarray.rpad(1, 0)) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], [], [0.0, 1.1, 2.2]]
    assert ak.to_list(indexedarray.rpad(2, 1)) == [[6.6, 7.7, 8.8, 9.9], [5.5, None], [3.3, 4.4], [None, None], [0.0, 1.1, 2.2]]
    assert ak.to_list(indexedarray.rpad(3, 1)) == [[6.6, 7.7, 8.8, 9.9], [5.5, None, None], [3.3, 4.4, None], [None, None, None], [0.0, 1.1, 2.2]]
    assert ak.to_list(indexedarray.rpad(4, 0)) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], [], [0.0, 1.1, 2.2]]
    assert ak.to_list(indexedarray.rpad(5, 0)) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], [], [0.0, 1.1, 2.2]]
    assert ak.to_list(indexedarray.rpad(6, 0)) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], [], [0.0, 1.1, 2.2], None]
    assert ak.to_list(indexedarray.rpad(7, 0)) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], [], [0.0, 1.1, 2.2], None, None]

def test_rpad_and_clip_indexed_array():
    listoffsetarray = ak.from_iter([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False)
    backward = ak.from_iter([[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], [], [0.0, 1.1, 2.2]], highlevel=False)

    index = ak.layout.Index64(np.array([4, 3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, listoffsetarray)
    assert ak.to_list(indexedarray) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], [], [0.0, 1.1, 2.2]]

    assert ak.to_list(backward.rpad_and_clip(4, 1)) == ak.to_list(indexedarray.rpad_and_clip(4, 1))
    assert ak.to_list(indexedarray.rpad_and_clip(1, 0)) == [[6.6, 7.7, 8.8, 9.9]]
    assert ak.to_list(indexedarray.rpad_and_clip(2, 0)) == [[6.6, 7.7, 8.8, 9.9], [5.5]]
    assert ak.to_list(indexedarray.rpad_and_clip(3, 0)) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4]]
    assert ak.to_list(indexedarray.rpad_and_clip(4, 0)) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], []]
    assert ak.to_list(indexedarray.rpad_and_clip(5, 0)) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], [], [0.0, 1.1, 2.2]]
    assert ak.to_list(indexedarray.rpad_and_clip(6, 0)) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], [], [0.0, 1.1, 2.2], None]
    assert ak.to_list(indexedarray.rpad_and_clip(7, 0)) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], [], [0.0, 1.1, 2.2], None, None]
    assert ak.to_list(indexedarray.rpad_and_clip(8, 0)) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], [], [0.0, 1.1, 2.2], None, None, None]

    assert ak.to_list(indexedarray.rpad_and_clip(1, 1)) == [[6.6], [5.5], [3.3], [None], [0.0]]
    assert ak.to_list(indexedarray.rpad_and_clip(2, 1)) == [[6.6, 7.7], [5.5, None], [3.3, 4.4], [None, None], [0.0, 1.1]]
    assert ak.to_list(indexedarray.rpad_and_clip(3, 1)) == [[6.6, 7.7, 8.8], [5.5, None, None], [3.3, 4.4, None], [None, None, None], [0.0, 1.1, 2.2]]
    assert ak.to_list(indexedarray.rpad_and_clip(4, 1)) == [[6.6, 7.7, 8.8, 9.9], [5.5, None, None, None], [3.3, 4.4, None, None], [None, None, None, None], [0.0, 1.1, 2.2, None]]
    assert ak.to_list(indexedarray.rpad_and_clip(5, 1)) == [[6.6, 7.7, 8.8, 9.9, None], [5.5, None, None, None, None], [3.3, 4.4, None, None, None], [None, None, None, None, None], [0.0, 1.1, 2.2, None, None]]

def test_rpad_indexed_option_array():
    listoffsetarray = ak.from_iter([[0.0, None, None], None, [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False)
    backward = ak.from_iter([[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], None, [0.0, None, None]], highlevel=False)

    index = ak.layout.Index64(np.array([4, 3, 2, -1, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedOptionArray64(index, listoffsetarray)
    assert ak.to_list(indexedarray) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], None, [0.0, None, None]]

    assert ak.to_list(backward.rpad(4, 1)) == ak.to_list(indexedarray.rpad(4, 1))
    assert ak.to_list(indexedarray.rpad(1, 0)) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], None, [0.0, None, None]]
    assert ak.to_list(indexedarray.rpad(1, 1)) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], None, [0.0, None, None]]
    assert ak.to_list(indexedarray.rpad(3, 1)) == [[6.6, 7.7, 8.8, 9.9], [5.5, None, None], [3.3, 4.4, None], None, [0.0, None, None]]
    assert ak.to_list(indexedarray.rpad(4, 0)) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], None, [0.0, None, None]]
    assert ak.to_list(indexedarray.rpad(5, 0)) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], None, [0.0, None, None]]
    assert ak.to_list(indexedarray.rpad(6, 0)) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], None, [0.0, None, None], None]
    assert ak.to_list(indexedarray.rpad(7, 0)) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], None, [0.0, None, None], None, None]
    assert ak.to_list(indexedarray.rpad(8, 0)) == [[6.6, 7.7, 8.8, 9.9], [5.5], [3.3, 4.4], None, [0.0, None, None], None, None, None]

    assert ak.to_list(indexedarray.rpad_and_clip(1, 0)) == [[6.6, 7.7, 8.8, 9.9]]
    assert ak.to_list(indexedarray.rpad_and_clip(1, 1)) == [[6.6], [5.5], [3.3], None, [0.0]]

def test_rpad_recordarray():
    array = ak.from_iter([{"x": [], "y": [2, 2]}, {"x": [1.1], "y": [1]}, {"x": [2.2, 2.2], "y": []}], highlevel=False)

    assert ak.to_list(array.rpad(5, 0)) == [{"x": [], "y": [2, 2]}, {"x": [1.1], "y": [1]}, {"x": [2.2, 2.2], "y": []}, None, None]

    assert ak.to_list(array.rpad(2, 1)) == [{"x": [None, None], "y": [2, 2]}, {"x": [1.1, None], "y": [1, None]}, {"x": [2.2, 2.2], "y": [None, None]}]

def test_rpad_unionarray():
    content1 = ak.from_iter([[], [1.1], [2.2, 2.2]], highlevel=False)
    content2 = ak.from_iter([[2, 2], [1], []], highlevel=False)
    tags = ak.layout.Index8(np.array([0, 1, 0, 1, 0, 1], dtype=np.int8))
    index = ak.layout.Index64(np.array([0, 0, 1, 1, 2, 2], dtype=np.int64))
    array = ak.layout.UnionArray8_64(tags, index, [content1, content2])
    assert ak.to_list(array) == [[], [2, 2], [1.1], [1], [2.2, 2.2], []]

    assert ak.to_list(array.rpad(7, 0)) == [[], [2, 2], [1.1], [1], [2.2, 2.2], [], None]

    assert ak.to_list(array.rpad(2, 1)) == [[None, None], [2, 2], [1.1, None], [1, None], [2.2, 2.2], [None, None]]
