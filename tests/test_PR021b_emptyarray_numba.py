# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import os
import json

import pytest
import numpy

import awkward1

pytest.skip("Disabling Numba until the rewrite is done.", allow_module_level=True)

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
