# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_record():
    array1 = awkward1.Array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]).layout
    assert awkward1.tolist(array1) == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]

    array2 = array1.setitem_field("z", awkward1.Array([[], [1], [2, 2]]).layout)
    assert awkward1.tolist(array2) == [{"x": 1, "y": 1.1, "z": []}, {"x": 2, "y": 2.2, "z": [1]}, {"x": 3, "y": 3.3, "z": [2, 2]}]

    array3 = array1.setitem_field(None, awkward1.Array([[], [1], [2, 2]]).layout)
    assert awkward1.tolist(array3) == [{"x": 1, "y": 1.1, "2": []}, {"x": 2, "y": 2.2, "2": [1]}, {"x": 3, "y": 3.3, "2": [2, 2]}]

    array3 = array1.setitem_field(0, awkward1.Array([[], [1], [2, 2]]).layout)
    assert awkward1.tolist(array3) == [{"x": 1, "y": 1.1, "0": []}, {"x": 2, "y": 2.2, "0": [1]}, {"x": 3, "y": 3.3, "0": [2, 2]}]

    array1 = awkward1.Array([(1, 1.1), (2, 2.2), (3, 3.3)]).layout
    assert awkward1.tolist(array1) == [(1, 1.1), (2, 2.2), (3, 3.3)]

    array2 = array1.setitem_field("z", awkward1.Array([[], [1], [2, 2]]).layout)
    assert awkward1.tolist(array2) == [{"0": 1, "1": 1.1, "z": []}, {"0": 2, "1": 2.2, "z": [1]}, {"0": 3, "1": 3.3, "z": [2, 2]}]

    array3 = array1.setitem_field(None, awkward1.Array([[], [1], [2, 2]]).layout)
    assert awkward1.tolist(array3) == [(1, 1.1, []), (2, 2.2, [1]), (3, 3.3, [2, 2])]

    array3 = array1.setitem_field(0, awkward1.Array([[], [1], [2, 2]]).layout)
    assert awkward1.tolist(array3) == [([], 1, 1.1), ([1], 2, 2.2), ([2, 2], 3, 3.3)]

    array3 = array1.setitem_field(1, awkward1.Array([[], [1], [2, 2]]).layout)
    assert awkward1.tolist(array3) == [(1, [], 1.1), (2, [1], 2.2), (3, [2, 2], 3.3)]

    array3 = array1.setitem_field(100, awkward1.Array([[], [1], [2, 2]]).layout)
    assert awkward1.tolist(array3) == [(1, 1.1, []), (2, 2.2, [1]), (3, 3.3, [2, 2])]
