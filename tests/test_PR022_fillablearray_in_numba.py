# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import os
import json

import pytest
import numpy

import awkward1

numba = pytest.importorskip("numba")

def test_boxing():
    @numba.njit
    def f1(q):
        b = q
        return 3.14

    a = awkward1.layout.FillableArray()
    assert sys.getrefcount(a) == 2
    f1(a)
    assert sys.getrefcount(a) == 2

    @numba.njit
    def f2(q):
        return q

    a = awkward1.layout.FillableArray()
    assert sys.getrefcount(a) == 2
    f2(a)
    assert sys.getrefcount(a) == 2
    b = f2(a)
    assert sys.getrefcount(a) == 3

    assert str(b.snapshot()) == "<EmptyArray/>"
