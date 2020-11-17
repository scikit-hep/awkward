# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test_jagged_axis0():
    assert awkward1.min(awkward1.Array([[1.1, 5.5], [4.4], [2.2, 3.3, 0.0, -10]]), axis=0).tolist() == [1.1, 3.3, 0, -10]
    assert awkward1.argmin(awkward1.Array([[1.1, 5.5], [4.4], [2.2, 3.3, 0.0, -10]]), axis=0).tolist() == [0, 2, 2, 2]

def test_jagged_axis1():
    # first is [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]

    array = awkward1.Array([[[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]], [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[4, 3, 2], [4, 3, 2]]

    array = awkward1.Array([[[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]], [[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[4, 3, 2], [5, 4, 3]]

    array = awkward1.Array([[[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]], [[], [], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[4, 3, 2], [6, 5, 4]]

    array = awkward1.Array([[[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]], [[1.1], [1.1, 2.2], [], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[4, 3, 2], [5, 4, 3]]

    array = awkward1.Array([[[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]], [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[4, 3, 2], [5, 4, 2]]

    array = awkward1.Array([[[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]], [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[4, 3, 2], [5, 3, 2]]

    array = awkward1.Array([[[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]], [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0], []]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[4, 3, 2], [4, 3, 2]]

    array = awkward1.Array([[[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]], [[1.1, 999, 999], [1.1, 2.2, 999], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[4, 3, 2], [4, 3, 2]]

    array = awkward1.Array([[[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]], [[1.1, 999, 999, 999], [1.1, 2.2, 999], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3, 999]]
    assert awkward1.argmin(array, axis=1).tolist() == [[4, 3, 2], [4, 3, 2, 0]]

    # first is [[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]

    array = awkward1.Array([[[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]], [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[5, 4, 3], [4, 3, 2]]

    array = awkward1.Array([[[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]], [[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[5, 4, 3], [5, 4, 3]]

    array = awkward1.Array([[[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]], [[], [], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[5, 4, 3], [6, 5, 4]]

    array = awkward1.Array([[[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]], [[1.1], [1.1, 2.2], [], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[5, 4, 3], [5, 4, 3]]

    array = awkward1.Array([[[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]], [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[5, 4, 3], [5, 4, 2]]

    array = awkward1.Array([[[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]], [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[5, 4, 3], [5, 3, 2]]

    array = awkward1.Array([[[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]], [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0], []]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[5, 4, 3], [4, 3, 2]]

    array = awkward1.Array([[[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]], [[1.1, 999, 999], [1.1, 2.2, 999], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[5, 4, 3], [4, 3, 2]]

    array = awkward1.Array([[[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]], [[1.1, 999, 999, 999], [1.1, 2.2, 999], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3, 999]]
    assert awkward1.argmin(array, axis=1).tolist() == [[5, 4, 3], [4, 3, 2, 0]]

    # first is [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]]

    array = awkward1.Array([[[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]], [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[5, 4, 2], [4, 3, 2]]

    array = awkward1.Array([[[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]], [[], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[5, 4, 2], [5, 4, 3]]

    array = awkward1.Array([[[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]], [[], [], [1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[5, 4, 2], [6, 5, 4]]

    array = awkward1.Array([[[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]], [[1.1], [1.1, 2.2], [], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[5, 4, 2], [5, 4, 3]]

    array = awkward1.Array([[[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]], [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[5, 4, 2], [5, 4, 2]]

    array = awkward1.Array([[[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]], [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[5, 4, 2], [5, 3, 2]]

    array = awkward1.Array([[[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]], [[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [999, 2.0], [1.0], []]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[5, 4, 2], [4, 3, 2]]

    array = awkward1.Array([[[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]], [[1.1, 999, 999], [1.1, 2.2, 999], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3]]
    assert awkward1.argmin(array, axis=1).tolist() == [[5, 4, 2], [4, 3, 2]]

    array = awkward1.Array([[[1.1], [1.1, 2.2], [1.1, 2.2, 3.3], [], [999, 2.0], [1.0]], [[1.1, 999, 999, 999], [1.1, 2.2, 999], [1.1, 2.2, 3.3], [999, 2.0], [1.0]]])
    assert awkward1.min(array, axis=1).tolist() == [[1, 2, 3.3], [1, 2, 3.3, 999]]
    assert awkward1.argmin(array, axis=1).tolist() == [[5, 4, 2], [4, 3, 2, 0]]

def test_IndexedOptionArray():
    content = awkward1.Array([1.1, 2.2, 3.3, 4.4, 5.5]).layout
    index = awkward1.layout.Index64(numpy.array([4, 2, -1, -1, 1, 0, 1]))
    array = awkward1.Array(awkward1.layout.IndexedOptionArray64(index, content))
    assert array.tolist() == [5.5, 3.3, None, None, 2.2, 1.1, 2.2]
    assert awkward1.min(array, axis=0) == 1.1
    assert awkward1.argmin(array, axis=0) == 5

    assert awkward1.argmin(awkward1.Array([[2.2, 1.1], [None, 3.3], [2.2, 1.1]]), axis=-1).tolist() == [1, 1, 1]
    assert awkward1.argmin(awkward1.Array([[2.2, 1.1], [None, 3.3], [2.2, None, 1.1]]), axis=-1).tolist() == [1, 1, 2]
    assert awkward1.argmin(awkward1.Array([[2.2, 1.1], [3.3, None], [2.2, None, 1.1]]), axis=-1).tolist() == [1, 0, 2]

    assert awkward1.argmin(awkward1.Array([[2.2, 1.1, 0.0], [], [None, 0.5], [2, 1]]), axis=0).tolist() == [3, 2, 0]
    assert awkward1.argmin(awkward1.Array([[2.2, 1.1, 0.0], [], [0.5, None], [2, 1]]), axis=0).tolist() == [2, 3, 0]
    assert awkward1.argmin(awkward1.Array([[2.2, 1.1, 0.0], [0.5, None], [], [2, 1]]), axis=0).tolist() == [1, 3, 0]

def test_ByteMaskedArray():
    content = awkward1.Array([1.1, 2.2, 3.3, 999, 999, 4.4, 5.5]).layout
    mask = awkward1.layout.Index8(numpy.array([False, False, False, True, True, False, False]))
    bytemaskedarray = awkward1.layout.ByteMaskedArray(mask, content, valid_when=False)
    array = awkward1.Array(bytemaskedarray)
    assert array.tolist() == [1.1, 2.2, 3.3, None, None, 4.4, 5.5]
    assert awkward1.max(array, axis=0) == 5.5
    assert awkward1.argmax(array, axis=0) == 6

    offsets = awkward1.layout.Index64(numpy.array([0, 2, 4, 7], dtype=numpy.int64))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, bytemaskedarray)
    array = awkward1.Array(listoffsetarray)
    assert array.tolist() == [[1.1, 2.2], [3.3, None], [None, 4.4, 5.5]]
    assert awkward1.max(array, axis=1).tolist() == [2.2, 3.3, 5.5]
    assert awkward1.argmax(array, axis=1).tolist() == [1, 0, 2]
