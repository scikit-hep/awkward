# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_num_1():
    content = awkward1.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]).layout
    bitmask = awkward1.layout.IndexU8(numpy.array([40, 34], dtype=numpy.uint8))
    array = awkward1.Array(awkward1.layout.BitMaskedArray(bitmask, content, False, 9, False))
    cuda_array = awkward1.copy_to(array, "cuda")
    assert awkward1.num(cuda_array, 0) == awkward1.num(array, 0)
    assert awkward1.num(cuda_array, 1) == awkward1.num(array, 1)

def test_num_2():
    content = awkward1.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]).layout
    bytemask = awkward1.layout.Index8(
        numpy.array([False, True, False], dtype=numpy.bool))
    array = awkward1.Array(awkward1.layout.ByteMaskedArray(bytemask, content, True))
    cuda_array = awkward1.copy_to(array, "cuda")
    assert awkward1.num(cuda_array, 0) == awkward1.num(array, 0)
    assert awkward1.num(cuda_array, 1) == awkward1.num(array, 1)

def test_num_3():
    array = awkward1.Array(awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])))
    cuda_array = awkward1.copy_to(array, "cuda")
    assert awkward1.num(cuda_array, 0) == awkward1.num(array, 0)

def test_num_4():
    array = awkward1.Array(awkward1.layout.NumpyArray(
        numpy.array([[0.0, 1.1], [2.2, 3.3], [4.4, 5.5]])))
    cuda_array = awkward1.copy_to(array, "cuda")
    assert awkward1.num(cuda_array, 0) == awkward1.num(array, 0)
    assert awkward1.num(cuda_array, 1).tolist() == awkward1.num(array, 1).tolist()

def test_num_5():
    array = awkward1.Array(awkward1.layout.EmptyArray())
    cuda_array = awkward1.copy_to(array, "cuda")
    assert awkward1.num(cuda_array, 0) == awkward1.num(array, 0)

def test_num_6():
    content = awkward1.layout.NumpyArray(
        numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    array = awkward1.Array(awkward1.layout.ListOffsetArray64(offsets, content))
    cuda_array = awkward1.copy_to(array, "cuda")
    assert awkward1.num(cuda_array, 0) == awkward1.num(array, 0)
    assert awkward1.num(cuda_array, 1).tolist() == awkward1.num(array, 1).tolist()

def test_num_7():
    content = awkward1.layout.NumpyArray(
        numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.IndexU32(numpy.array([0, 3, 3, 5, 6, 9]))
    array = awkward1.Array(awkward1.layout.ListOffsetArrayU32(offsets, content))
    cuda_array = awkward1.copy_to(array, "cuda")
    assert awkward1.num(cuda_array, 0) == awkward1.num(array, 0)
    assert awkward1.num(cuda_array, 1).tolist() == awkward1.num(array, 1).tolist()

def test_num_8():
    content = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
    regulararray = awkward1.layout.RegularArray(listoffsetarray, 2)
    starts = awkward1.layout.Index64(numpy.array([0, 1]))
    stops = awkward1.layout.Index64(numpy.array([2, 3]))
    listarray = awkward1.layout.ListArray64(starts, stops, regulararray)

    cuda_listoffsetarray = awkward1.copy_to(listoffsetarray, "cuda")
    assert awkward1.num(cuda_listoffsetarray, 0) == awkward1.num(awkward1.Array(listoffsetarray), 0)
    assert awkward1.num(cuda_listoffsetarray, 1).tolist() == awkward1.num(awkward1.Array(listoffsetarray), 1).tolist()

    cuda_regulararray = awkward1.copy_to(regulararray, "cuda")
    assert awkward1.num(cuda_regulararray, 0) == awkward1.num(awkward1.Array(regulararray), 0)
    assert awkward1.num(cuda_regulararray, 1).tolist() == awkward1.num(awkward1.Array(regulararray), 1).tolist()

    cuda_listarray = awkward1.copy_to(listarray, "cuda")
    assert awkward1.num(cuda_listarray, 0) == awkward1.num(awkward1.Array(listarray), 0)
    assert awkward1.num(cuda_listarray, 1).tolist() == awkward1.num(awkward1.Array(listarray), 1).tolist()

    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
    content2 = awkward1.layout.NumpyArray(
        numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index32(numpy.array([0, 3, 3, 5, 6, 9]))

    recordarray = awkward1.Array(awkward1.layout.RecordArray(
        [content1, listoffsetarray, content2, content1], keys=["one", "two", "2", "wonky"]))

    cuda_recordarray = awkward1.copy_to(recordarray, "cuda")
    assert awkward1.num(cuda_recordarray, 0).tolist() == awkward1.num(recordarray, 0).tolist()

    content0 = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    content = awkward1.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]).layout
    tags = awkward1.layout.Index8(
        numpy.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=numpy.int8))
    index = awkward1.layout.Index32(
        numpy.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=numpy.int32))
    unionarray = awkward1.Array(awkward1.layout.UnionArray8_32(
        tags, index, [content0, content1]))

    cuda_unionarray = awkward1.copy_to(unionarray, "cuda")
    assert awkward1.num(cuda_unionarray, 0) == awkward1.num(unionarray, 0)

def test_num_9():
    content = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.Index32(
        numpy.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=numpy.int64))
    indexedarray = awkward1.Array(awkward1.layout.IndexedArray32(index, content))

    cuda_indexedarray = awkward1.copy_to(indexedarray, "cuda")
    assert awkward1.num(cuda_indexedarray, 0) == awkward1.num(indexedarray, 0)


    ioa = awkward1.Array(awkward1.layout.IndexedOptionArray32(
              awkward1.layout.Index32([-30, 19, 6, 7, -3, 21, 13, 22, 17, 9, -12, 16]),
              awkward1.layout.NumpyArray(
                  numpy.array([5.2, 1.7, 6.7, -0.4, 4.0, 7.8, 3.8, 6.8, 4.2, 0.3, 4.6, 6.2,
                               6.9, -0.7, 3.9, 1.6, 8.7, -0.7, 3.2, 4.3, 4.0, 5.8, 4.2, 7.0,
                               5.6, 3.8]))))
    cuda_ioa = awkward1.copy_to(ioa, "cuda")
    copyback_ioa = awkward1.copy_to(cuda_ioa, "cpu")
    assert awkward1.num(cuda_ioa, 0) == awkward1.num(ioa, 0)
