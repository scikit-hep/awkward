# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_sort_emptyarray():
    array = awkward1.layout.EmptyArray()
    assert awkward1.tolist(array.sort(-1, True, False)) == None

def test_sort_numpyarray():
    array = awkward1.layout.NumpyArray(numpy.array([3, 2, 1, 5, 4]))
    assert awkward1.tolist(array.sort(-1, True, False)) == [2, 1, 0, 4, 3]
    # assert awkward1.tolist(array.sort(-1, False)) == [3, 4, 0, 2, 1]

    # np.sort([[3.3, 2.2, 4.4], [1.1, 5.5, 3.3]], axis=-1)
    # array([[2.2, 3.3, 4.4],
    #        [1.1, 3.3, 5.5]])

    # np.sort([[3.3, 2.2, 4.4], [1.1, 5.5, 3.3]], axis=-2)
    # array([[1.1, 2.2, 3.3],
    #        [3.3, 5.5, 4.4]])
    array = awkward1.layout.NumpyArray(numpy.array([[2.2, 1.1, 3.3], [6.6, 4.4, 5.5]]))
    #assert awkward1.tolist(array.sort(-1, True, False)) == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]
    assert awkward1.tolist(array.sort(-2, True, False)) == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]
