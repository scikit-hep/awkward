# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak

def test_num_1():
    content = ak.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]).layout
    bitmask = ak.layout.IndexU8(np.array([40, 34], dtype=np.uint8))
    array = ak.Array(ak.layout.BitMaskedArray(bitmask, content, False, 9, False))
    cuda_array = ak.to_kernels(array, "cuda")
    assert ak.num(cuda_array, 0) == ak.num(array, 0)
    assert ak.num(cuda_array, 1) == ak.num(array, 1)

def test_num_2():
    content = ak.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]).layout
    bytemask = ak.layout.Index8(
        np.array([False, True, False], dtype=np.bool))
    array = ak.Array(ak.layout.ByteMaskedArray(bytemask, content, True))
    cuda_array = ak.to_kernels(array, "cuda")
    assert ak.num(cuda_array, 0) == ak.num(array, 0)
    assert ak.num(cuda_array, 1) == ak.num(array, 1)

def test_num_3():
    array = ak.Array(ak.layout.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])))
    cuda_array = ak.to_kernels(array, "cuda")
    assert ak.num(cuda_array, 0) == ak.num(array, 0)

def test_num_4():
    array = ak.Array(ak.layout.NumpyArray(
        np.array([[0.0, 1.1], [2.2, 3.3], [4.4, 5.5]])))
    cuda_array = ak.to_kernels(array, "cuda")
    assert ak.num(cuda_array, 0) == ak.num(array, 0)
    assert ak.num(cuda_array, 1).tolist() == ak.num(array, 1).tolist()

def test_num_5():
    array = ak.Array(ak.layout.EmptyArray())
    cuda_array = ak.to_kernels(array, "cuda")
    assert ak.num(cuda_array, 0) == ak.num(array, 0)

def test_num_6():
    content = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 9]))
    array = ak.Array(ak.layout.ListOffsetArray64(offsets, content))
    cuda_array = ak.to_kernels(array, "cuda")
    assert ak.num(cuda_array, 0) == ak.num(array, 0)
    assert ak.num(cuda_array, 1).tolist() == ak.num(array, 1).tolist()

def test_num_7():
    content = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = ak.layout.IndexU32(np.array([0, 3, 3, 5, 6, 9]))
    array = ak.Array(ak.layout.ListOffsetArrayU32(offsets, content))
    cuda_array = ak.to_kernels(array, "cuda")
    assert ak.num(cuda_array, 0) == ak.num(array, 0)
    assert ak.num(cuda_array, 1).tolist() == ak.num(array, 1).tolist()

def test_num_8():
    content = ak.layout.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10]))
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)
    regulararray = ak.layout.RegularArray(listoffsetarray, 2)
    starts = ak.layout.Index64(np.array([0, 1]))
    stops = ak.layout.Index64(np.array([2, 3]))
    listarray = ak.layout.ListArray64(starts, stops, regulararray)

    cuda_listoffsetarray = ak.to_kernels(listoffsetarray, "cuda")
    assert ak.num(cuda_listoffsetarray, 0) == ak.num(ak.Array(listoffsetarray), 0)
    assert ak.num(cuda_listoffsetarray, 1).tolist() == ak.num(ak.Array(listoffsetarray), 1).tolist()

    cuda_regulararray = ak.to_kernels(regulararray, "cuda")
    assert ak.num(cuda_regulararray, 0) == ak.num(ak.Array(regulararray), 0)
    assert ak.num(cuda_regulararray, 1).tolist() == ak.num(ak.Array(regulararray), 1).tolist()

    cuda_listarray = ak.to_kernels(listarray, "cuda")
    assert ak.num(cuda_listarray, 0) == ak.num(ak.Array(listarray), 0)
    assert ak.num(cuda_listarray, 1).tolist() == ak.num(ak.Array(listarray), 1).tolist()

    content1 = ak.layout.NumpyArray(np.array([1, 2, 3, 4, 5]))
    content2 = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = ak.layout.Index32(np.array([0, 3, 3, 5, 6, 9]))

    recordarray = ak.Array(ak.layout.RecordArray(
        [content1, listoffsetarray, content2, content1], keys=["one", "two", "2", "wonky"]))

    cuda_recordarray = ak.to_kernels(recordarray, "cuda")
    assert ak.num(cuda_recordarray, 0).tolist() == ak.num(recordarray, 0).tolist()

    content0 = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    content = ak.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]).layout
    tags = ak.layout.Index8(
        np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))
    index = ak.layout.Index32(
        np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    unionarray = ak.Array(ak.layout.UnionArray8_32(
        tags, index, [content0, content1]))

    cuda_unionarray = ak.to_kernels(unionarray, "cuda")
    assert ak.num(cuda_unionarray, 0) == ak.num(unionarray, 0)

def test_num_9():
    content = ak.layout.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = ak.layout.Index32(
        np.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=np.int64))
    indexedarray = ak.Array(ak.layout.IndexedArray32(index, content))

    cuda_indexedarray = ak.to_kernels(indexedarray, "cuda")
    assert ak.num(cuda_indexedarray, 0) == ak.num(indexedarray, 0)


    ioa = ak.Array(ak.layout.IndexedOptionArray32(
              ak.layout.Index32([-30, 19, 6, 7, -3, 21, 13, 22, 17, 9, -12, 16]),
              ak.layout.NumpyArray(
                  np.array([5.2, 1.7, 6.7, -0.4, 4.0, 7.8, 3.8, 6.8, 4.2, 0.3, 4.6, 6.2,
                               6.9, -0.7, 3.9, 1.6, 8.7, -0.7, 3.2, 4.3, 4.0, 5.8, 4.2, 7.0,
                               5.6, 3.8]))))
    cuda_ioa = ak.to_kernels(ioa, "cuda")
    copyback_ioa = ak.to_kernels(cuda_ioa, "cpu")
    assert ak.num(cuda_ioa, 0) == ak.num(ioa, 0)
