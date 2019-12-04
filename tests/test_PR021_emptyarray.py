# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import os
import json

import pytest
import numpy

import awkward1

def test_unknown():
    a = awkward1.fromjson("[[], [], []]").layout
    assert awkward1.tolist(a) == [[], [], []]
    assert str(awkward1.typeof(a)) == "3 * var * unknown"
    assert awkward1.typeof(a) == awkward1.layout.ArrayType(awkward1.layout.ListType(awkward1.layout.UnknownType()), 3)
    assert not awkward1.typeof(a) == awkward1.layout.ArrayType(awkward1.layout.PrimitiveType("float64"), 3)

    a = awkward1.fromjson("[[], [[], []], [[], [], []]]").layout
    assert awkward1.tolist(a) == [[], [[], []], [[], [], []]]
    assert str(awkward1.typeof(a)) == "3 * var * var * unknown"
    assert awkward1.typeof(a) == awkward1.layout.ArrayType(awkward1.layout.ListType(awkward1.layout.ListType(awkward1.layout.UnknownType())), 3)

    a = awkward1.layout.FillableArray()
    a.beginlist()
    a.endlist()
    a.beginlist()
    a.endlist()
    a.beginlist()
    a.endlist()
    assert awkward1.tolist(a) == [[], [], []]
    assert str(awkward1.typeof(a)) == "3 * var * unknown"
    assert awkward1.typeof(a) == awkward1.layout.ArrayType(awkward1.layout.ListType(awkward1.layout.UnknownType()), 3)
    assert not awkward1.typeof(a) == awkward1.layout.ArrayType(awkward1.layout.PrimitiveType("float64"), 3)

    a = a.snapshot()
    assert awkward1.tolist(a) == [[], [], []]
    assert str(awkward1.typeof(a)) == "3 * var * unknown"
    assert awkward1.typeof(a) == awkward1.layout.ArrayType(awkward1.layout.ListType(awkward1.layout.UnknownType()), 3)
    assert not awkward1.typeof(a) == awkward1.layout.ArrayType(awkward1.layout.PrimitiveType("float64"), 3)

def test_getitem():
    a = awkward1.fromjson("[[], [[], []], [[], [], []]]")
    assert awkward1.tolist(a[2]) == [[], [], []]

    assert awkward1.tolist(a[2, 1]) == []
    with pytest.raises(ValueError) as excinfo:
        a[2, 1, 0]
    assert str(excinfo.value) == "in ListArray64 attempting to get 0, index out of range"
    assert awkward1.tolist(a[2, 1][()]) == []
    with pytest.raises(ValueError) as excinfo:
        a[2, 1][0]
    assert str(excinfo.value) == "in EmptyArray attempting to get 0, index out of range"
    assert awkward1.tolist(a[2, 1][100:200]) == []
    assert awkward1.tolist(a[2, 1, 100:200]) == []
    assert awkward1.tolist(a[2, 1][numpy.array([], dtype=int)]) == []
    assert awkward1.tolist(a[2, 1, numpy.array([], dtype=int)]) == []
    with pytest.raises(ValueError) as excinfo:
        a[2, 1, numpy.array([0], dtype=int)]
    assert str(excinfo.value) == "in ListArray64 attempting to get 0, index out of range"
    with pytest.raises(ValueError) as excinfo:
        a[2, 1][100:200, 0]
    assert str(excinfo.value) == "in EmptyArray, too many dimensions in slice"
    with pytest.raises(ValueError) as excinfo:
        a[2, 1][100:200, 200:300]
    assert str(excinfo.value) == "in EmptyArray, too many dimensions in slice"
    with pytest.raises(ValueError) as excinfo:
        a[2, 1][100:200, numpy.array([], dtype=int)]
    assert str(excinfo.value) == "in EmptyArray, too many dimensions in slice"

    assert awkward1.tolist(a[1:, 1:]) == [[[]], [[], []]]
    with pytest.raises(ValueError) as excinfo:
        a[1:, 1:, 0]
    assert str(excinfo.value) == "in ListArray64 attempting to get 0, index out of range"

numba = pytest.importorskip("numba")
def test_numba():
    a = awkward1.fromjson("[[], [[], []], [[], [], []]]").layout

    @numba.njit
    def f1(q):
        return q[2, 1]
    assert awkward1.tolist(f1(a)) == []

    @numba.njit
    def f2(q):
        return q[2, 1][()]
    assert awkward1.tolist(f2(a)) == []

    @numba.njit
    def f3(q):
        return q[2, 1][100:200]
    assert awkward1.tolist(f3(a)) == []

    @numba.njit
    def f4(q):
        return q[2, 1, 0]
    with pytest.raises(numba.errors.TypingError):
        f4(a)

    @numba.njit
    def f5(q):
        return q[2, 1, 100:200]
    assert awkward1.tolist(f5(a)) == []

    @numba.njit
    def f6a(q):
        return q[2, 1, 100:200, 0]
    with pytest.raises(numba.errors.TypingError):
        f6a(a)

    @numba.njit
    def f6b(q):
        return q[2, 1, 100:200][0]
    with pytest.raises(numba.errors.TypingError):
        f6b(a)

    @numba.njit
    def f7a(q):
        return q[2, 1, 100:200, 200:300]
    with pytest.raises(numba.errors.TypingError):
        f7a(a)

    @numba.njit
    def f7b(q):
        return q[2, 1, 100:200][200:300]
    assert awkward1.tolist(f7b(a)) == []

    @numba.njit
    def f7c(q):
        return q[2, 1, 100:200][()]
    assert awkward1.tolist(f7c(a)) == []

    @numba.njit
    def f8a(q):
        return q[2, 1, 100:200, numpy.array([], dtype=numpy.int64)]
    with pytest.raises(numba.errors.TypingError):
        f8a(a)

    @numba.njit
    def f8b(q, z):
        return q[2, 1, z]
    assert awkward1.tolist(f8b(a, numpy.array([], dtype=int))) == []

    @numba.njit
    def f8c(q, z):
        return q[2, 1, z, z]
    with pytest.raises(numba.errors.TypingError):
        f8c(a, numpy.array([], dtype=int))

    @numba.njit
    def f8d(q, z):
        return q[2, 1, z][()]
    assert awkward1.tolist(f8d(a, numpy.array([], dtype=int))) == []
