# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_ByteMaskedArray():
    content = awkward1.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]).layout
    mask = awkward1.layout.Index8(numpy.array([0, 0, 1, 1, 0], dtype=numpy.int8))
    array = awkward1.layout.ByteMaskedArray(mask, content, validwhen=False)
    assert awkward1.tolist(array) == [[0.0, 1.1, 2.2], [], None, None, [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.tolist(array[-1]) == [6.6, 7.7, 8.8, 9.9]
    assert awkward1.tolist(array[-2]) == None
    assert awkward1.tolist(array[1:]) == [[], None, None, [6.6, 7.7, 8.8, 9.9]]
    assert array[4, 1] == 7.7
    assert awkward1.tolist(array[2:, 1]) == [None, None, 7.7]
    assert awkward1.tolist(array[2:, [2, 1, 1, 0]]) == [None, None, [8.8, 7.7, 7.7, 6.6]]

    content = awkward1.Array([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}]).layout
    mask = awkward1.layout.Index8(numpy.array([0, 0, 1, 1, 0], dtype=numpy.int8))
    array = awkward1.layout.ByteMaskedArray(mask, content, validwhen=False)
    assert awkward1.tolist(array) == [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, None, None, {"x": 4.4, "y": [4, 4, 4, 4]}]
    assert awkward1.tolist(array["x"]) == [0.0, 1.1, None, None, 4.4]
    assert awkward1.tolist(array[["x", "y"]]) == [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, None, None, {"x": 4.4, "y": [4, 4, 4, 4]}]

def test_ByteMaskedArray_jaggedslice0():
    array = awkward1.Array([[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], checkvalid=True).layout
    index = awkward1.layout.Index64(numpy.array([0, 1, 2, 3], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedOptionArray64(index, array)
    assert awkward1.tolist(indexedarray) == [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.tolist(indexedarray[awkward1.Array([[0, -1], [0], [], [1, 1]])]) == [[0.0, 2.2], [3.3], [], [7.7, 7.7]]

    mask = awkward1.layout.Index8(numpy.array([0, 0, 0, 0], dtype=numpy.int8))
    maskedarray = awkward1.layout.ByteMaskedArray(mask, array, validwhen=False)
    assert awkward1.tolist(maskedarray) == [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.tolist(maskedarray[awkward1.Array([[0, -1], [0], [], [1, 1]])]) == [[0.0, 2.2], [3.3], [], [7.7, 7.7]]

def test_ByteMaskedArray_jaggedslice1():
    model = awkward1.Array([[0.0, 1.1, None, 2.2], [], [3.3, None, 4.4], [5.5], [6.6, 7.7, None, 8.8, 9.9]])
    assert awkward1.tolist(model[awkward1.Array([[3, 2, 1, 1, 0], [], [1], [0, 0], [1, 2]])]) == [[2.2, None, 1.1, 1.1, 0.0], [], [None], [5.5, 5.5], [7.7, None]]

    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 999, 2.2, 3.3, 123, 4.4, 5.5, 6.6, 7.7, 321, 8.8, 9.9]))
    mask = awkward1.layout.Index8(numpy.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=numpy.int8))
    maskedarray = awkward1.layout.ByteMaskedArray(mask, content, validwhen=False)
    offsets = awkward1.layout.Index64(numpy.array([0, 4, 4, 7, 8, 13], dtype=numpy.int64))
    listarray = awkward1.Array(awkward1.layout.ListOffsetArray64(offsets, maskedarray))
    assert awkward1.tolist(listarray) == awkward1.tolist(model)
    assert awkward1.tolist(listarray[awkward1.Array([[3, 2, 1, 1, 0], [], [1], [0, 0], [1, 2]])]) == [[2.2, None, 1.1, 1.1, 0.0], [], [None], [5.5, 5.5], [7.7, None]]

def test_ByteMaskedArray_jaggedslice2():
    model = awkward1.Array([[[0.0, 1.1, None, 2.2], [], [3.3, None, 4.4]], [], [[5.5]], [[6.6, 7.7, None, 8.8, 9.9]]])
    assert awkward1.tolist(model[awkward1.Array([[[3, 2, 1, 1, 0], [], [1]], [], [[0, 0]], [[1, 2]]])]) == [[[2.2, None, 1.1, 1.1, 0.0], [], [None]], [], [[5.5, 5.5]], [[7.7, None]]]

    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 999, 2.2, 3.3, 123, 4.4, 5.5, 6.6, 7.7, 321, 8.8, 9.9]))
    mask = awkward1.layout.Index8(numpy.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=numpy.int8))
    maskedarray = awkward1.layout.ByteMaskedArray(mask, content, validwhen=False)
    offsets = awkward1.layout.Index64(numpy.array([0, 4, 4, 7, 8, 13], dtype=numpy.int64))
    sublistarray = awkward1.layout.ListOffsetArray64(offsets, maskedarray)
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 3, 4, 5], dtype=numpy.int64))
    listarray = awkward1.Array(awkward1.layout.ListOffsetArray64(offsets2, sublistarray))
    assert awkward1.tolist(listarray) == awkward1.tolist(model)
    assert awkward1.tolist(listarray[awkward1.Array([[[3, 2, 1, 1, 0], [], [1]], [], [[0, 0]], [[1, 2]]])]) == [[[2.2, None, 1.1, 1.1, 0.0], [], [None]], [], [[5.5, 5.5]], [[7.7, None]]]

def test_ByteMaskedArray_jaggedslice3():
    model = awkward1.Array([[[[0.0, 1.1, None, 2.2], [], [3.3, None, 4.4]], []], [[[5.5]], [[6.6, 7.7, None, 8.8, 9.9]]]])
    assert awkward1.tolist(model[awkward1.Array([[[[3, 2, 1, 1, 0], [], [1]], []], [[[0, 0]], [[1, 2]]]])]) == [[[[2.2, None, 1.1, 1.1, 0.0], [], [None]], []], [[[5.5, 5.5]], [[7.7, None]]]]

    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 999, 2.2, 3.3, 123, 4.4, 5.5, 6.6, 7.7, 321, 8.8, 9.9]))
    mask = awkward1.layout.Index8(numpy.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=numpy.int8))
    maskedarray = awkward1.layout.ByteMaskedArray(mask, content, validwhen=False)
    offsets = awkward1.layout.Index64(numpy.array([0, 4, 4, 7, 8, 13], dtype=numpy.int64))
    subsublistarray = awkward1.layout.ListOffsetArray64(offsets, maskedarray)
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 3, 4, 5], dtype=numpy.int64))
    sublistarray = awkward1.layout.ListOffsetArray64(offsets2, subsublistarray)
    offsets3 = awkward1.layout.Index64(numpy.array([0, 2, 4], dtype=numpy.int64))
    listarray = awkward1.Array(awkward1.layout.ListOffsetArray64(offsets3, sublistarray))
    assert awkward1.tolist(listarray) == awkward1.tolist(model)
    assert awkward1.tolist(listarray[awkward1.Array([[[[3, 2, 1, 1, 0], [], [1]], []], [[[0, 0]], [[1, 2]]]])]) == [[[[2.2, None, 1.1, 1.1, 0.0], [], [None]], []], [[[5.5, 5.5]], [[7.7, None]]]]

def test_ByteMaskedArray_to_slice():
    content = awkward1.layout.NumpyArray(numpy.array([5, 2, 999, 3, 9, 123, 1], dtype=numpy.int64))
    mask = awkward1.layout.Index8(numpy.array([0, 0, 1, 0, 0, 1, 0], dtype=numpy.int8))
    maskedarray = awkward1.layout.ByteMaskedArray(mask, content, validwhen=False)
    assert awkward1.tolist(maskedarray) == [5, 2, None, 3, 9, None, 1]

    assert awkward1.layout._slice_tostring(maskedarray) == "[missing([0, 1, -1, ..., 3, -1, 4], array([5, 2, 3, 9, 1]))]"

def test_ByteMaskedArray_as_slice():
    array = awkward1.Array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], checkvalid=True)
    slicecontent = awkward1.Array([3, 6, 999, 123, -2, 6]).layout
    slicemask = awkward1.layout.Index8(numpy.array([0, 0, 1, 1, 0, 0], dtype=numpy.int8))
    slicearray = awkward1.layout.ByteMaskedArray(slicemask, slicecontent, validwhen=False)

    assert awkward1.tolist(array[slicearray]) == [3.3, 6.6, None, None, 8.8, 6.6]
