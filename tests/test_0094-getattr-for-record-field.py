# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_field_names():
    array = awkward1.Array([[{"x": 1.1, "y": [1], "z": "one"}, {"x": 2.2, "y": [2, 2], "z": "two"}, {"x": 3.3, "y": [3, 3, 3], "z": "three"}], [], [{"x": 4.4, "y": [4, 4, 4, 4], "z": "four"}, {"x": 5.5, "y": [5, 5, 5, 5, 5], "z": "five"}]], check_valid=True)
    assert "x" in dir(array)
    assert "y" in dir(array)
    assert "z" in dir(array)

    assert awkward1.to_list(array.x) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert awkward1.to_list(array.y) == [[[1], [2, 2], [3, 3, 3]], [], [[4, 4, 4, 4], [5, 5, 5, 5, 5]]]
    assert awkward1.to_list(array.z) == [["one", "two", "three"], [], ["four", "five"]]

def test_tuple_ids():
    array = awkward1.Array([[(1.1, [1], "one"), (2.2, [2, 2], "two"), (3.3, [3, 3, 3], "three")], [], [(4.4, [4, 4, 4, 4], "four"), (5.5, [5, 5, 5, 5, 5], "five")]], check_valid=True)

    assert awkward1.to_list(array.slot0) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert awkward1.to_list(array.slot1) == [[[1], [2, 2], [3, 3, 3]], [], [[4, 4, 4, 4], [5, 5, 5, 5, 5]]]
    assert awkward1.to_list(array.slot2) == [["one", "two", "three"], [], ["four", "five"]]
