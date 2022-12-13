# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import sys

import numpy as np  # noqa: F401
import pytest

import awkward as ak

numba = pytest.importorskip("numba")

ak.numba.register_and_check()


def test_refcount():
    array = ak.highlevel.Array([1, 2, 3])

    @numba.njit
    def f1():
        array
        return 3.14

    @numba.njit
    def f2():
        array, array
        return 3.14

    assert sys.getrefcount(array) == 2
    f1()
    assert sys.getrefcount(array) == 3
    f2()
    assert sys.getrefcount(array) == 5

    del f1
    assert sys.getrefcount(array) == 4


def test_Array():
    array = ak.highlevel.Array([1, 2, 3])

    @numba.njit
    def f1():
        array
        return 3.14

    f1()
    assert (
        sys.getrefcount(array._numbaview),
        sys.getrefcount(array._numbaview.lookup),
    ) == (2, 2)

    @numba.njit
    def f2():
        return array

    a = f2()
    assert a.to_list() == [1, 2, 3]
    assert (
        sys.getrefcount(array._numbaview),
        sys.getrefcount(array._numbaview.lookup),
    ) == (2, 2)

    @numba.njit
    def f3():
        return array, array

    b, c = f3()
    assert b.to_list() == [1, 2, 3]
    assert c.to_list() == [1, 2, 3]
    assert (
        sys.getrefcount(array._numbaview),
        sys.getrefcount(array._numbaview.lookup),
    ) == (2, 2)

    del a
    assert (
        sys.getrefcount(array._numbaview),
        sys.getrefcount(array._numbaview.lookup),
    ) == (2, 2)

    del b
    assert (
        sys.getrefcount(array._numbaview),
        sys.getrefcount(array._numbaview.lookup),
    ) == (2, 2)

    del c
    assert (
        sys.getrefcount(array._numbaview),
        sys.getrefcount(array._numbaview.lookup),
    ) == (2, 2)

    @numba.njit
    def f4():
        return array[1]

    assert f4() == 2
    assert (
        sys.getrefcount(array._numbaview),
        sys.getrefcount(array._numbaview.lookup),
    ) == (2, 2)


def test_Record():
    record = ak.Record({"x": 1, "y": [1, 2, 3]})

    @numba.njit
    def f1():
        return record.y[1]

    assert f1() == 2


def test_ArrayBuilder():
    builder = ak.highlevel.ArrayBuilder()
    assert sys.getrefcount(builder._layout) == 3

    @numba.njit
    def f():
        builder.append(1)
        builder.append(2)
        builder.append(3)
        return builder, builder

    @numba.njit
    def g():
        builder.append(1)
        builder.append(2)
        builder.append(3)

    b, c = f()
    assert b.snapshot().to_list() == [1, 2, 3]
    assert c.snapshot().to_list() == [1, 2, 3]
    assert builder.snapshot().to_list() == [1, 2, 3]

    assert sys.getrefcount(builder._layout) == 5

    g()
    assert b.snapshot().to_list() == [1, 2, 3, 1, 2, 3]
    assert c.snapshot().to_list() == [1, 2, 3, 1, 2, 3]
    assert builder.snapshot().to_list() == [1, 2, 3, 1, 2, 3]

    assert sys.getrefcount(builder._layout) == 5

    del b._layout
    assert sys.getrefcount(builder._layout) == 4

    del c._layout
    assert sys.getrefcount(builder._layout) == 3
