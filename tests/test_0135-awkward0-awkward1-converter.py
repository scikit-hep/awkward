# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

awkward0 = pytest.importorskip("awkward")

def test_toawkward0():
    array = awkward1.Array([1.1, 2.2, 3.3, 4.4])
    assert isinstance(awkward1.toawkward0(array), numpy.ndarray)
    assert awkward1.toawkward0(array).tolist() == [1.1, 2.2, 3.3, 4.4]

    array = awkward1.Array(numpy.arange(2*3*5).reshape(2, 3, 5)).layout.toRegularArray()
    assert isinstance(awkward1.toawkward0(array), awkward0.JaggedArray)
    assert awkward1.toawkward0(array).tolist() == [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]], [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]

    array = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    assert isinstance(awkward1.toawkward0(array), awkward0.JaggedArray)
    assert awkward1.toawkward0(array).tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    array = awkward1.layout.ListArray64(awkward1.layout.Index64(numpy.array([4, 999, 1], dtype=numpy.int64)), awkward1.layout.Index64(numpy.array([7, 999, 3], dtype=numpy.int64)), awkward1.layout.NumpyArray(numpy.array([3.14, 4.4, 5.5, 123, 1.1, 2.2, 3.3, 321])))
    assert isinstance(awkward1.toawkward0(array), awkward0.JaggedArray)
    assert awkward1.toawkward0(array).tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    array = awkward1.Array([{"x": 0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}])
    assert isinstance(awkward1.toawkward0(array[2]), dict)
    assert awkward1.toawkward0(array[2])["x"] == 2.2
    assert isinstance(awkward1.toawkward0(array[2])["y"], numpy.ndarray)
    assert awkward1.toawkward0(array[2])["y"].tolist() == [2, 2]

    assert isinstance(awkward1.toawkward0(array), awkward0.Table)
    assert awkward1.toawkward0(array).tolist() == [{"x": 0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}]

    array = awkward1.Array([(0, []), (1.1, [1]), (2.2, [2, 2]), (3.3, [3, 3, 3])])
    assert isinstance(awkward1.toawkward0(array), awkward0.Table)
    assert awkward1.toawkward0(array).tolist() == [(0, []), (1.1, [1]), (2.2, [2, 2]), (3.3, [3, 3, 3])]

    array = awkward1.Array([0.0, [], 1.1, [1], 2.2, [2, 2], 3.3, [3, 3, 3]])
    assert isinstance(awkward1.toawkward0(array), awkward0.UnionArray)
    assert awkward1.toawkward0(array).tolist() == [0.0, [], 1.1, [1], 2.2, [2, 2], 3.3, [3, 3, 3]]

    array = awkward1.Array([1.1, 2.2, None, None, 3.3, None, 4.4])
    assert isinstance(awkward1.toawkward0(array), awkward0.IndexedMaskedArray)
    assert awkward1.toawkward0(array).tolist() == [1.1, 2.2, None, None, 3.3, None, 4.4]

    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.Index64(numpy.array([3, 2, 2, 5, 0], dtype=numpy.int64))
    array = awkward1.layout.IndexedArray64(index, content)
    assert isinstance(awkward1.toawkward0(array), awkward0.IndexedArray)
    assert awkward1.toawkward0(array).tolist() == [3.3, 2.2, 2.2, 5.5, 0.0]
