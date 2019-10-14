# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools

import pytest
import numpy

import awkward1

def test_boolean():
    a = awkward1.layout.FillableArray()
    a.boolean(True)
    a.boolean(True)
    a.boolean(False)
    a.boolean(True)
    assert awkward1.tolist(a.snapshot()) == [True, True, False, True]
    assert awkward1.tolist(a) == [True, True, False, True]
    assert awkward1.tolist(a[1:-1]) == [True, False]
