# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test():
    data = awkward1.Array([{"x": i} for i in range(10)])
    y = awkward1.Array(numpy.array([[i, i] for i in range(10)]))
    data["y"] = y
    assert data.tolist() == [{"x": 0, "y": [0, 0]}, {"x": 1, "y": [1, 1]}, {"x": 2, "y": [2, 2]}, {"x": 3, "y": [3, 3]}, {"x": 4, "y": [4, 4]}, {"x": 5, "y": [5, 5]}, {"x": 6, "y": [6, 6]}, {"x": 7, "y": [7, 7]}, {"x": 8, "y": [8, 8]}, {"x": 9, "y": [9, 9]}]
