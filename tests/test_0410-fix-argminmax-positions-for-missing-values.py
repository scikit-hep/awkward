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
