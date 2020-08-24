# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_tocuda():
    content = awkward1.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]).layout
    bitmask = awkward1.layout.IndexU8(numpy.array([40, 34], dtype=numpy.uint8))
    array = awkward1.layout.BitMaskedArray(bitmask, content, False, 9, False)
    cuda_array = awkward1.copy_to(array, "cuda")
    copyback_array = awkward1.copy_to(cuda_array, "cpu")
    assert awkward1.to_list(cuda_array) == awkward1.to_list(array)
    assert awkward1.to_list(copyback_array) == awkward1.to_list(array)

    bytemask = awkward1.layout.Index8(
    numpy.array([False, True, False], dtype=numpy.bool))
    array = awkward1.layout.ByteMaskedArray(bytemask, content, True)
    cuda_array = awkward1.copy_to(array, "cuda")
    copyback_array = awkward1.copy_to(cuda_array, "cpu")
    assert awkward1.to_list(cuda_array) == awkward1.to_list(array)
    assert awkward1.to_list(copyback_array) == awkward1.to_list(array)

    array = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5]))
    cuda_array = awkward1.copy_to(array, "cuda")
    copyback_array = awkward1.copy_to(cuda_array, "cpu")
    assert awkward1.to_list(cuda_array) == awkward1.to_list(array)
    assert awkward1.to_list(copyback_array) == awkward1.to_list(array)

    array = awkward1.layout.NumpyArray(
        numpy.array([[0.0, 1.1], [2.2, 3.3], [4.4, 5.5]]))
    cuda_array = awkward1.copy_to(array, "cuda")
    copyback_array = awkward1.copy_to(cuda_array, "cpu")

    assert awkward1.to_list(cuda_array) == awkward1.to_list(array)
    assert awkward1.to_list(copyback_array) == awkward1.to_list(array)

    array = awkward1.layout.EmptyArray()
    cuda_array = awkward1.copy_to(array, "cuda")
    copyback_array = awkward1.copy_to(cuda_array, "cpu")
    assert awkward1.to_list(cuda_array) == awkward1.to_list(array)
    assert awkward1.to_list(copyback_array) == awkward1.to_list(array)

    content = awkward1.layout.NumpyArray(
        numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    array = awkward1.layout.ListOffsetArray64(offsets, content)
    cuda_array = awkward1.copy_to(array, "cuda")
    copyback_array = awkward1.copy_to(cuda_array, "cpu")
    assert awkward1.to_list(cuda_array) == awkward1.to_list(array)
    assert awkward1.to_list(copyback_array) == awkward1.to_list(array)

    content = awkward1.layout.NumpyArray(
        numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.IndexU32(numpy.array([0, 3, 3, 5, 6, 9]))
    array = awkward1.layout.ListOffsetArrayU32(offsets, content)
    cuda_array = awkward1.copy_to(array, "cuda")
    copyback_array = awkward1.copy_to(cuda_array, "cpu")
    assert awkward1.to_list(cuda_array) == awkward1.to_list(array)
    assert awkward1.to_list(copyback_array) == awkward1.to_list(array)

    # # Testing parameters
    content = awkward1.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]).layout
    offsets = awkward1.layout.Index32(numpy.array([0, 3, 3, 5, 6, 9]))
    array = awkward1.layout.ListOffsetArray32(offsets, content)
    cuda_array = awkward1.copy_to(array, "cuda")
    copyback_array = awkward1.copy_to(cuda_array, "cpu")
    assert awkward1.to_list(cuda_array) == awkward1.to_list(array)
    assert awkward1.to_list(copyback_array) == awkward1.to_list(array)

    content = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
    regulararray = awkward1.layout.RegularArray(listoffsetarray, 2)
    starts = awkward1.layout.Index64(numpy.array([0, 1]))
    stops = awkward1.layout.Index64(numpy.array([2, 3]))
    listarray = awkward1.layout.ListArray64(starts, stops, regulararray)

    cuda_listoffsetarray = awkward1.copy_to(listoffsetarray, "cuda")
    copyback_listoffsetarray = awkward1.copy_to(cuda_listoffsetarray, "cpu")
    assert awkward1.to_list(cuda_listoffsetarray) == awkward1.to_list(listoffsetarray)
    assert awkward1.to_list(copyback_listoffsetarray) == awkward1.to_list(listoffsetarray)

    cuda_regulararray = awkward1.copy_to(regulararray, "cuda")
    copyback_regulararray = awkward1.copy_to(cuda_regulararray, "cpu")
    assert awkward1.to_list(cuda_regulararray) == awkward1.to_list(regulararray)
    assert awkward1.to_list(copyback_regulararray) == awkward1.to_list(regulararray)

    cuda_listarray = awkward1.copy_to(listarray, "cuda")
    copyback_listarray = awkward1.copy_to(cuda_listarray, "cpu")
    assert awkward1.to_list(cuda_listarray) == awkward1.to_list(listarray)
    assert awkward1.to_list(copyback_listarray) == awkward1.to_list(listarray)

    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
    content2 = awkward1.layout.NumpyArray(
        numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index32(numpy.array([0, 3, 3, 5, 6, 9]))

    recordarray = awkward1.layout.RecordArray(
        [content1, listoffsetarray, content2, content1], keys=["one", "two", "2", "wonky"])

    cuda_recordarray = awkward1.copy_to(recordarray, "cuda")
    copyback_recordarray = awkward1.copy_to(cuda_recordarray, "cpu")
    assert awkward1.to_list(cuda_recordarray) == awkward1.to_list(recordarray)
    assert awkward1.to_list(copyback_recordarray) == awkward1.to_list(recordarray)

    content0 = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    content = awkward1.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]).layout
    tags = awkward1.layout.Index8(
        numpy.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=numpy.int8))
    index = awkward1.layout.Index32(
        numpy.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=numpy.int32))
    unionarray = awkward1.layout.UnionArray8_32(
        tags, index, [content0, content1])

    cuda_unionarray = awkward1.copy_to(unionarray, "cuda")
    copyback_unionarray = awkward1.copy_to(cuda_unionarray, "cpu")
    assert awkward1.to_list(cuda_unionarray) == awkward1.to_list(unionarray)
    assert awkward1.to_list(copyback_unionarray) == awkward1.to_list(unionarray)

    content = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.Index32(
        numpy.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray32(index, content)

    cuda_indexedarray = awkward1.copy_to(indexedarray, "cuda")
    copyback_indexedarray = awkward1.copy_to(cuda_indexedarray, "cpu")
    assert awkward1.to_list(cuda_indexedarray) == awkward1.to_list(indexedarray)
    assert awkward1.to_list(copyback_indexedarray) == awkward1.to_list(indexedarray)

    bytemaskedarray = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array(
        [True, True, False, False, False], dtype=numpy.int8)), listoffsetarray, True)
    cuda_bytemaskedarray = awkward1.copy_to(bytemaskedarray, "cuda")
    copyback_bytemaskedarray = awkward1.copy_to(cuda_bytemaskedarray, "cpu")
    assert awkward1.to_list(cuda_bytemaskedarray) == awkward1.to_list(bytemaskedarray)
    assert awkward1.to_list(copyback_bytemaskedarray) == awkward1.to_list(bytemaskedarray)


    bytemaskedarray = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(
        numpy.array([True, False], dtype=numpy.int8)), listarray, True)
    cuda_bytemaskedarray = awkward1.copy_to(bytemaskedarray, "cuda")
    copyback_bytemaskedarray = awkward1.copy_to(cuda_bytemaskedarray, "cpu")
    assert awkward1.to_list(cuda_bytemaskedarray) == awkward1.to_list(bytemaskedarray)
    assert awkward1.to_list(copyback_bytemaskedarray) == awkward1.to_list(bytemaskedarray)


    bytemaskedarray = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(
        numpy.array([True, False], dtype=numpy.int8)), recordarray, True)
    cuda_bytemaskedarray = awkward1.copy_to(bytemaskedarray, "cuda")
    copyback_bytemaskedarray = awkward1.copy_to(cuda_bytemaskedarray, "cpu")
    assert awkward1.to_list(cuda_bytemaskedarray) == awkward1.to_list(bytemaskedarray)
    assert awkward1.to_list(copyback_bytemaskedarray) == awkward1.to_list(bytemaskedarray)


    bytemaskedarray = awkward1.layout.ByteMaskedArray(
        awkward1.layout.Index8(numpy.array([True, False, False], dtype=numpy.int8)), indexedarray, True)
    cuda_bytemaskedarray = awkward1.copy_to(bytemaskedarray, "cuda")
    copyback_bytemaskedarray = awkward1.copy_to(cuda_bytemaskedarray, "cpu")
    assert awkward1.to_list(cuda_bytemaskedarray) == awkward1.to_list(bytemaskedarray)
    assert awkward1.to_list(copyback_bytemaskedarray) == awkward1.to_list(bytemaskedarray)

    bytemaskedarray = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(
        numpy.array([True, False, False], dtype=numpy.int8)), unionarray, True)
    cuda_bytemaskedarray = awkward1.copy_to(bytemaskedarray, "cuda")
    copyback_bytemaskedarray = awkward1.copy_to(cuda_bytemaskedarray, "cpu")
    assert awkward1.to_list(cuda_bytemaskedarray) == awkward1.to_list(bytemaskedarray)
    assert awkward1.to_list(copyback_bytemaskedarray) == awkward1.to_list(bytemaskedarray)

    ioa = awkward1.layout.IndexedOptionArray32(
        awkward1.layout.Index32([-30, 19, 6, 7, -3, 21, 13, 22, 17, 9, -12, 16]),
        awkward1.layout.NumpyArray(numpy.array([5.2, 1.7, 6.7, -0.4, 4.0, 7.8, 3.8, 6.8, 4.2, 0.3, 4.6, 6.2,
                                                6.9, -0.7, 3.9, 1.6, 8.7, -0.7, 3.2, 4.3, 4.0, 5.8, 4.2, 7.0,
                                                5.6, 3.8])))
    cuda_ioa = awkward1.copy_to(ioa, "cuda")
    copyback_ioa = awkward1.copy_to(cuda_ioa, "cpu")
    assert awkward1.to_list(cuda_ioa) == awkward1.to_list(ioa)
    assert awkward1.to_list(copyback_ioa) == awkward1.to_list(ioa)
