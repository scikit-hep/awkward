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
    assert awkward1.tolist(array[2:, 1]) == [None, None, 7.7]

    content = awkward1.Array([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}]).layout
    mask = awkward1.layout.Index8(numpy.array([0, 0, 1, 1, 0], dtype=numpy.int8))
    array = awkward1.layout.ByteMaskedArray(mask, content, validwhen=False)
    assert awkward1.tolist(array) == [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, None, None, {"x": 4.4, "y": [4, 4, 4, 4]}]
    assert awkward1.tolist(array["x"]) == [0.0, 1.1, None, None, 4.4]
    assert awkward1.tolist(array[["x", "y"]]) == [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, None, None, {"x": 4.4, "y": [4, 4, 4, 4]}]

    # raise Exception
