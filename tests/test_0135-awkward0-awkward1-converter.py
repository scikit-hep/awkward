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
