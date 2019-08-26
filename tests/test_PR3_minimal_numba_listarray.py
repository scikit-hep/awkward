# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy
numba = pytest.importorskip("numba")

import awkward1

def test_numpyarray_boxing(capsys):
    a = numpy.arange(10)
    wrapped = awkward1.layout.NumpyArray(a)
    assert (sys.getrefcount(a), sys.getrefcount(wrapped)) == (3, 2)
    @numba.njit
    def f1(q):
        return q
    out = f1(wrapped)
    out2 = [f1(wrapped) for i in range(100)]
    assert (sys.getrefcount(a), sys.getrefcount(wrapped)) == (3, 3 + 100)
    assert numpy.asarray(out).tolist() == list(range(10))
    del out
    del out2
    assert (sys.getrefcount(a), sys.getrefcount(wrapped)) == (3, 2)

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

def test_dummy1():
    a = awkward1.layout.NumpyArray(numpy.array([5, 4, 3, 2, 1], dtype="i4"))
    @numba.njit
    def f1(q):
        return q.dummy1()
    assert f1(a) == 25

def test_listoffsetarray_boxing():
    offsets = awkward1.layout.Index(numpy.array([0, 2, 2, 3], "i4"))
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]))
    array = awkward1.layout.ListOffsetArray(offsets, content)
    assert (sys.getrefcount(offsets), sys.getrefcount(content), sys.getrefcount(array)) == (2, 2, 2)

    @numba.njit
    def f1(q):
        return q
    out = f1(array)
    assert (sys.getrefcount(offsets), sys.getrefcount(content), sys.getrefcount(array)) == (2, 2, 2)
    assert numpy.asarray(out.offsets()).tolist() == [0, 2, 2, 3]
    assert numpy.asarray(out.content()).tolist() == [1.1, 2.2, 3.3]

    array2 = awkward1.layout.ListOffsetArray(offsets, array)
    out2 = f1(array2)
    assert (sys.getrefcount(offsets), sys.getrefcount(content), sys.getrefcount(array)) == (2, 2, 2)
    assert numpy.asarray(out2.offsets()).tolist() == [0, 2, 2, 3]
    assert numpy.asarray(out2.content().offsets()).tolist() == [0, 2, 2, 3]
    assert numpy.asarray(out2.content().content()).tolist() == [1.1, 2.2, 3.3]
