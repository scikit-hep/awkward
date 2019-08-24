# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy
numba = pytest.importorskip("numba")

import awkward1

def test_boxing():
    a = awkward1.layout.NumpyArray(numpy.arange(10))

    @numba.njit
    def f1(q):
        return q

    f1(a)

def test_len():
    a = awkward1.layout.NumpyArray(numpy.arange(10))

    @numba.njit
    def f1(q):
        return len(q)

    assert f1(a) == 10

def test_getitem_int():
    a = awkward1.layout.NumpyArray(numpy.arange(12).reshape(3, 4))

    @numba.njit
    def f1(q):
        return q[1]
    out = f1(a)
    assert isinstance(out, awkward1.layout.NumpyArray)
    assert numpy.asarray(out).tolist() == [4, 5, 6, 7]

    @numba.njit
    def f2(q):
        return q[1][2]
    assert f2(a) == 6

    @numba.njit
    def f3(q):
        return q[1, 2]
    assert f3(a) == 6

def test_getitem_slice():
    a = awkward1.layout.NumpyArray(numpy.arange(12).reshape(3, 4))

    @numba.njit
    def f1(q):
        return q[1:]
    out = f1(a)
    assert isinstance(out, awkward1.layout.NumpyArray)
    assert numpy.asarray(out).tolist() == [[4, 5, 6, 7], [8, 9, 10, 11]]
