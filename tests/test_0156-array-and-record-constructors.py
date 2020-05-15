# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_record():
    assert awkward1.to_list(awkward1.Record({"x": 1, "y": 2.2})) == {"x": 1, "y": 2.2}

def test_fromiter():
    array = awkward1.Array([numpy.array([1, 2, 3]), numpy.array([4, 5, 6, 7])])

    assert str(awkward1.type(array)) == "2 * var * int64"
    assert awkward1.to_list(array) == [[1, 2, 3], [4, 5, 6, 7]]
