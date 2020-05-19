# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

numexpr = pytest.importorskip("numexpr")

def test_numexpr():
    a = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)
    b = awkward1.Array([100, 200, 300], check_valid=True)
    assert awkward1.to_list(awkward1.numexpr.evaluate("a + b")) == [[101.1, 102.2, 103.3], [], [304.4, 305.5]]
    a = [1, 2, 3]
    assert awkward1.to_list(awkward1.numexpr.re_evaluate()) == [101, 202, 303]

def test_broadcast_arrays():
    a = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)
    b = awkward1.Array([100, 200, 300], check_valid=True)

    out = awkward1.broadcast_arrays(a, b)
    assert awkward1.to_list(out[0]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert awkward1.to_list(out[1]) == [[100, 100, 100], [], [300, 300]]
