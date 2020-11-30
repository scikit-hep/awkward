# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy as np
import awkward1 as ak

py27 = (sys.version_info[0] < 3)

def test_index():
    array_i1 = np.array([np.iinfo("i1").min, -1, 0, 1, np.iinfo("i1").max], dtype="i1")
    array_u1 = np.array([np.iinfo("u1").min, -1, 0, 1, np.iinfo("u1").max], dtype="u1")
    array_li2 = np.array([np.iinfo("<i2").min, -1, 0, 1, np.iinfo("<i2").max], dtype="<i2")
    array_lu2 = np.array([np.iinfo("<u2").min, -1, 0, 1, np.iinfo("<u2").max], dtype="<u2")
    array_li4 = np.array([np.iinfo("<i4").min, -1, 0, 1, np.iinfo("<i4").max], dtype="<i4")
    array_lu4 = np.array([np.iinfo("<u4").min, -1, 0, 1, np.iinfo("<u4").max], dtype="<u4")
    array_li8 = np.array([np.iinfo("<i8").min, -1, 0, 1, np.iinfo("<i8").max], dtype="<i8")
    array_lu8 = np.array([np.iinfo("<u8").min, -1, 0, 1, np.iinfo("<u8").max], dtype="<u8")
    array_bi2 = np.array([np.iinfo(">i2").min, -1, 0, 1, np.iinfo(">i2").max], dtype=">i2")
    array_bu2 = np.array([np.iinfo(">u2").min, -1, 0, 1, np.iinfo(">u2").max], dtype=">u2")
    array_bi4 = np.array([np.iinfo(">i4").min, -1, 0, 1, np.iinfo(">i4").max], dtype=">i4")
    array_bu4 = np.array([np.iinfo(">u4").min, -1, 0, 1, np.iinfo(">u4").max], dtype=">u4")
    array_bi8 = np.array([np.iinfo(">i8").min, -1, 0, 1, np.iinfo(">i8").max], dtype=">i8")
    array_bu8 = np.array([np.iinfo(">u8").min, -1, 0, 1, np.iinfo(">u8").max], dtype=">u8")

    index_i1 = ak.layout.Index8(array_i1)
    index_u1 = ak.layout.IndexU8(array_u1)
    index_li2 = ak.layout.Index32(array_li2)
    index_lu2 = ak.layout.Index32(array_lu2)
    index_li4 = ak.layout.Index32(array_li4)
    index_lu4 = ak.layout.IndexU32(array_lu4)
    index_li8 = ak.layout.Index64(array_li8)
    index_lu8 = ak.layout.Index64(array_lu8)
    index_bi2 = ak.layout.Index32(array_bi2)
    index_bu2 = ak.layout.Index32(array_bu2)
    index_bi4 = ak.layout.Index32(array_bi4)
    index_bu4 = ak.layout.IndexU32(array_bu4)
    index_bi8 = ak.layout.Index64(array_bi8)
    index_bu8 = ak.layout.Index64(array_bu8)

    assert index_i1[2] == 0
    assert index_u1[2] == 0
    assert index_li2[2] == 0
    assert index_lu2[2] == 0
    assert index_li4[2] == 0
    assert index_lu4[2] == 0
    assert index_li8[2] == 0
    assert index_lu8[2] == 0
    assert index_bi2[2] == 0
    assert index_bu2[2] == 0
    assert index_bi4[2] == 0
    assert index_bu4[2] == 0
    assert index_bi8[2] == 0
    assert index_bu8[2] == 0

    array_i1[2] = 10
    array_u1[2] = 10
    array_li2[2] = 10
    array_lu2[2] = 10
    array_li4[2] = 10
    array_lu4[2] = 10
    array_li8[2] = 10
    array_lu8[2] = 10
    array_bi2[2] = 10
    array_bu2[2] = 10
    array_bi4[2] = 10
    array_bu4[2] = 10
    array_bi8[2] = 10
    array_bu8[2] = 10

    assert index_i1[2] == 10
    assert index_u1[2] == 10
    assert index_li2[2] == 0
    assert index_lu2[2] == 0
    assert index_li4[2] == 10
    assert index_lu4[2] == 10
    assert index_li8[2] == 10
    assert index_lu8[2] == 0
    assert index_bi2[2] == 0
    assert index_bu2[2] == 0
    assert index_bi4[2] == 0
    assert index_bu4[2] == 0
    assert index_bi8[2] == 0
    assert index_bu8[2] == 0

content  = ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
starts1  = ak.layout.IndexU32(np.array([0, 3, 3, 5, 6], np.uint32))
stops1   = ak.layout.IndexU32(np.array([3, 3, 5, 6, 9], np.uint32))
offsets1 = ak.layout.IndexU32(np.array([0, 3, 3, 5, 6, 9], np.uint32))
starts2  = ak.layout.IndexU32(np.array([0, 2, 3, 3], np.uint32))
stops2   = ak.layout.IndexU32(np.array([2, 3, 3, 5], np.uint32))
offsets2 = ak.layout.IndexU32(np.array([0, 2, 3, 3, 5], np.uint32))

def test_listarray_basic():
    array1 = ak.layout.ListArrayU32(starts1, stops1, content)
    array2 = ak.layout.ListArrayU32(starts2, stops2, array1)
    assert ak.to_list(array1) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert ak.to_list(array1[2]) == [4.4, 5.5]
    assert ak.to_list(array1[1:-1]) == [[], [4.4, 5.5], [6.6]]
    assert ak.to_list(array2) == [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9]]]
    assert ak.to_list(array2[1]) == [[4.4, 5.5]]
    assert ak.to_list(array2[1:-1]) == [[[4.4, 5.5]], []]

def test_listoffsetarray_basic():
    array1 = ak.layout.ListOffsetArrayU32(offsets1, content)
    array2 = ak.layout.ListOffsetArrayU32(offsets2, array1)
    assert ak.to_list(array1) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert ak.to_list(array1[2]) == [4.4, 5.5]
    assert ak.to_list(array1[1:-1]) == [[], [4.4, 5.5], [6.6]]
    assert ak.to_list(array2) == [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9]]]
    assert ak.to_list(array2[1]) == [[4.4, 5.5]]
    assert ak.to_list(array2[1:-1]) == [[[4.4, 5.5]], []]

def test_listarray_at():
    array1 = ak.layout.ListArrayU32(starts1, stops1, content)
    array2 = ak.layout.ListArrayU32(starts2, stops2, array1)
    assert ak.to_list(array1[2]) == [4.4, 5.5]
    assert ak.to_list(array1[2,]) == [4.4, 5.5]
    assert ak.to_list(array1[2, 1:]) == [5.5]
    assert ak.to_list(array1[2:, 0]) == [4.4, 6.6, 7.7]
    assert ak.to_list(array1[2:, -1]) == [5.5, 6.6, 9.9]

def test_listoffsetarray_at():
    array1 = ak.layout.ListOffsetArrayU32(offsets1, content)
    array2 = ak.layout.ListOffsetArrayU32(offsets2, array1)
    assert ak.to_list(array1[2,]) == [4.4, 5.5]
    assert ak.to_list(array1[2, 1:]) == [5.5]
    assert ak.to_list(array1[2:, 0]) == [4.4, 6.6, 7.7]
    assert ak.to_list(array1[2:, -1]) == [5.5, 6.6, 9.9]

def test_listarray_slice():
    array1 = ak.layout.ListArrayU32(starts1, stops1, content)
    array2 = ak.layout.ListArrayU32(starts2, stops2, array1)
    assert ak.to_list(array1[1:-1]) == [[], [4.4, 5.5], [6.6]]
    assert ak.to_list(array1[1:-1,]) == [[], [4.4, 5.5], [6.6]]
    assert ak.to_list(array2[1:-1]) == [[[4.4, 5.5]], []]
    assert ak.to_list(array2[1:-1,]) == [[[4.4, 5.5]], []]

def test_listoffsetarray_slice():
    array1 = ak.layout.ListOffsetArrayU32(offsets1, content)
    array2 = ak.layout.ListOffsetArrayU32(offsets2, array1)
    assert ak.to_list(array1[1:-1]) == [[], [4.4, 5.5], [6.6]]
    assert ak.to_list(array1[1:-1,]) == [[], [4.4, 5.5], [6.6]]
    assert ak.to_list(array2[1:-1]) == [[[4.4, 5.5]], []]
    assert ak.to_list(array2[1:-1,]) == [[[4.4, 5.5]], []]

def test_listarray_slice_slice():
    array1 = ak.layout.ListArrayU32(starts1, stops1, content)
    array2 = ak.layout.ListArrayU32(starts2, stops2, array1)
    assert ak.to_list(array1[2:]) == [[4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert ak.to_list(array1[2:, 1:]) == [[5.5], [], [8.8, 9.9]]
    assert ak.to_list(array1[2:,:-1]) == [[4.4], [], [7.7, 8.8]]

def test_listoffsetarray_slice_slice():
    array1 = ak.layout.ListOffsetArrayU32(offsets1, content)
    array2 = ak.layout.ListOffsetArrayU32(offsets2, array1)
    assert ak.to_list(array1[2:]) == [[4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert ak.to_list(array1[2:, 1:]) == [[5.5], [], [8.8, 9.9]]
    assert ak.to_list(array1[2:,:-1]) == [[4.4], [], [7.7, 8.8]]

def test_listarray_ellipsis():
    array1 = ak.layout.ListArrayU32(starts1, stops1, content)
    array2 = ak.layout.ListArrayU32(starts2, stops2, array1)
    if not py27:
        assert ak.to_list(array1[Ellipsis, 1:]) == [[2.2, 3.3], [], [5.5], [], [8.8, 9.9]]
        assert ak.to_list(array2[Ellipsis, 1:]) == [[[2.2, 3.3], []], [[5.5]], [], [[], [8.8, 9.9]]]

def test_listoffsetarray_ellipsis():
    array1 = ak.layout.ListOffsetArrayU32(offsets1, content)
    array2 = ak.layout.ListOffsetArrayU32(offsets2, array1)
    if not py27:
        assert ak.to_list(array1[Ellipsis, 1:]) == [[2.2, 3.3], [], [5.5], [], [8.8, 9.9]]
        assert ak.to_list(array2[Ellipsis, 1:]) == [[[2.2, 3.3], []], [[5.5]], [], [[], [8.8, 9.9]]]

def test_listarray_array_slice():
    array1 = ak.layout.ListArrayU32(starts1, stops1, content)
    array2 = ak.layout.ListArrayU32(starts2, stops2, array1)
    assert ak.to_list(array2[[0, 0, 1, 1, 1, 0]]) == [[[1.1, 2.2, 3.3], []], [[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [[4.4, 5.5]], [[4.4, 5.5]], [[1.1, 2.2, 3.3], []]]
    assert ak.to_list(array2[[0, 0, 1, 1, 1, 0], :]) == [[[1.1, 2.2, 3.3], []], [[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [[4.4, 5.5]], [[4.4, 5.5]], [[1.1, 2.2, 3.3], []]]
    assert ak.to_list(array2[[0, 0, 1, 1, 1, 0], :, 1:]) == [[[2.2, 3.3], []], [[2.2, 3.3], []], [[5.5]], [[5.5]], [[5.5]], [[2.2, 3.3], []]]

def test_listoffsetarray_array_slice():
    array1 = ak.layout.ListOffsetArrayU32(offsets1, content)
    array2 = ak.layout.ListOffsetArrayU32(offsets2, array1)
    assert ak.to_list(array2[[0, 0, 1, 1, 1, 0]]) == [[[1.1, 2.2, 3.3], []], [[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [[4.4, 5.5]], [[4.4, 5.5]], [[1.1, 2.2, 3.3], []]]
    assert ak.to_list(array2[[0, 0, 1, 1, 1, 0], :]) == [[[1.1, 2.2, 3.3], []], [[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [[4.4, 5.5]], [[4.4, 5.5]], [[1.1, 2.2, 3.3], []]]
    assert ak.to_list(array2[[0, 0, 1, 1, 1, 0], :, 1:]) == [[[2.2, 3.3], []], [[2.2, 3.3], []], [[5.5]], [[5.5]], [[5.5]], [[2.2, 3.3], []]]

def test_listarray_array():
    array1 = ak.layout.ListArrayU32(starts1, stops1, content)
    array2 = ak.layout.ListArrayU32(starts2, stops2, array1)
    assert ak.to_list(array1[np.array([2, 0, 0, 1, -1])]) == [[4.4, 5.5], [1.1, 2.2, 3.3], [1.1, 2.2, 3.3], [], [7.7, 8.8, 9.9]]
    assert ak.to_list(array1[np.array([2, 0, 0, -1]), np.array([1, 1, 0, 0])]) == [5.5, 2.2, 1.1, 7.7]

    content_deep = ak.layout.NumpyArray(np.array([[0, 0], [1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60], [7, 70], [8, 80]]))
    starts1_deep = ak.layout.IndexU32(np.array([0, 3, 6]))
    stops1_deep = ak.layout.IndexU32(np.array([3, 6, 9]))
    array1_deep = ak.layout.ListArrayU32(starts1_deep, stops1_deep, content_deep)

    assert ak.to_list(array1_deep) == [[[0, 0], [1, 10], [2, 20]], [[3, 30], [4, 40], [5, 50]], [[6, 60], [7, 70], [8, 80]]]
    s = (np.array([2, 0, 0, -1]), np.array([1, 1, 0, 0]), np.array([0, 1, 0, 1]))
    assert np.array([[[0, 0], [1, 10], [2, 20]], [[3, 30], [4, 40], [5, 50]], [[6, 60], [7, 70], [8, 80]]])[s].tolist() == ak.to_list(array1_deep[s])

    s = (np.array([2, 0, 0, -1]), np.array([1, 1, 0, 0]), slice(1, None))
    assert np.array([[[0, 0], [1, 10], [2, 20]], [[3, 30], [4, 40], [5, 50]], [[6, 60], [7, 70], [8, 80]]])[s].tolist() == ak.to_list(array1_deep[s])

def test_listoffsetarray_array():
    array1 = ak.layout.ListOffsetArrayU32(offsets1, content)
    array2 = ak.layout.ListOffsetArrayU32(offsets2, array1)
    assert ak.to_list(array1[np.array([2, 0, 0, 1, -1])]) == [[4.4, 5.5], [1.1, 2.2, 3.3], [1.1, 2.2, 3.3], [], [7.7, 8.8, 9.9]]
    assert ak.to_list(array1[np.array([2, 0, 0, -1]), np.array([1, 1, 0, 0])]) == [5.5, 2.2, 1.1, 7.7]

    content_deep = ak.layout.NumpyArray(np.array([[0, 0], [1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60], [7, 70], [8, 80]]))
    starts1_deep = ak.layout.IndexU32(np.array([0, 3, 6]))
    stops1_deep = ak.layout.IndexU32(np.array([3, 6, 9]))
    array1_deep = ak.layout.ListArrayU32(starts1_deep, stops1_deep, content_deep)

    assert ak.to_list(array1_deep) == [[[0, 0], [1, 10], [2, 20]], [[3, 30], [4, 40], [5, 50]], [[6, 60], [7, 70], [8, 80]]]
    s = (np.array([2, 0, 0, -1]), np.array([1, 1, 0, 0]), np.array([0, 1, 0, 1]))
    assert np.array([[[0, 0], [1, 10], [2, 20]], [[3, 30], [4, 40], [5, 50]], [[6, 60], [7, 70], [8, 80]]])[s].tolist() == ak.to_list(array1_deep[s])

    s = (np.array([2, 0, 0, -1]), np.array([1, 1, 0, 0]), slice(1, None))
    assert np.array([[[0, 0], [1, 10], [2, 20]], [[3, 30], [4, 40], [5, 50]], [[6, 60], [7, 70], [8, 80]]])[s].tolist() == ak.to_list(array1_deep[s])
