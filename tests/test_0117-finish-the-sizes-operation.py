# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_emptyarray():
    array = awkward1.layout.EmptyArray()
    assert awkward1.tolist(array.sizes(0)) == 0
    assert awkward1.tolist(array.sizes(1)) == []
    assert awkward1.tolist(array.sizes(2)) == []

def test_numpyarray():
    array = awkward1.layout.NumpyArray(numpy.arange(2*3*5*7).reshape(2, 3, 5, 7))
    assert array.sizes(0) == 2
    assert awkward1.tolist(array.sizes(1)) == [3, 3]
    assert awkward1.tolist(array.sizes(2)) == [[5, 5, 5], [5, 5, 5]]
    assert awkward1.tolist(array.sizes(3)) == [[[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]], [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]]]
    with pytest.raises(ValueError) as err:
        array.sizes(4)
    assert str(err.value) == "'axis' out of range for 'sizes'"
