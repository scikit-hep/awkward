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
        z = q
        return 3.14

    a = awkward1.layout.FillableArray()
    assert sys.getrefcount(a) == 2
    f1(a)
    assert sys.getrefcount(a) == 2

    @numba.njit
    def f2(q):
        z = q
        return q

    a = awkward1.layout.FillableArray()
    assert sys.getrefcount(a) == 2
    f2(a)
    assert sys.getrefcount(a) == 2
    b = f2(a)
    assert sys.getrefcount(a) == 3

    print(str(b.snapshot()))

    assert str(b.snapshot()) == """<EmptyArray>
    <type>unknown</type>
</EmptyArray>"""

def test_simple():
    @numba.njit
    def f1(q):
        q.clear()
        return 3.14

    a = awkward1.layout.FillableArray()
    f1(a)

def test_boolean():
    @numba.njit
    def f1(q):
        q.boolean(True)
        q.boolean(False)
        q.boolean(False)
        return q

    a = awkward1.layout.FillableArray()
    b = f1(a)
    assert awkward1.tolist(a.snapshot()) == [True, False, False]
    assert awkward1.tolist(b.snapshot()) == [True, False, False]

def test_integer():
    @numba.njit
    def f1(q):
        q.integer(1)
        q.integer(2)
        q.integer(3)
        return q

    a = awkward1.layout.FillableArray()
    b = f1(a)
    assert awkward1.tolist(a.snapshot()) == [1, 2, 3]
    assert awkward1.tolist(b.snapshot()) == [1, 2, 3]

def test_real():
    @numba.njit
    def f1(q, z):
        q.real(1)
        q.real(2.2)
        q.real(z)
        return q

    a = awkward1.layout.FillableArray()
    b = f1(a, numpy.array([3.5], dtype=numpy.float32)[0])
    assert awkward1.tolist(a.snapshot()) == [1, 2.2, 3.5]
    assert awkward1.tolist(b.snapshot()) == [1, 2.2, 3.5]

def test_list():
    @numba.njit
    def f1(q):
        q.beginlist()
        q.real(1.1)
        q.real(2.2)
        q.real(3.3)
        q.endlist()
        q.beginlist()
        q.endlist()
        q.beginlist()
        q.real(4.4)
        q.real(5.5)
        q.endlist()
        return q

    a = awkward1.layout.FillableArray()
    b = f1(a)
    assert awkward1.tolist(a.snapshot()) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert awkward1.tolist(b.snapshot()) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    @numba.njit
    def f2(q):
        return len(q)

    assert f2(a) == 3
    assert f2(b) == 3

    @numba.njit
    def f3(q):
        q.clear()
        return q

    c = f3(b)
    assert awkward1.tolist(a.snapshot()) == []
    assert awkward1.tolist(b.snapshot()) == []
    assert awkward1.tolist(c.snapshot()) == []
