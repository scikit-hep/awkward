# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy
numba = pytest.importorskip("numba")

import awkward1

def test_iterator():
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]))
    offsets = awkward1.layout.Index32(numpy.array([0, 2, 2, 3], "i4"))
    array = awkward1.layout.ListOffsetArray32(offsets, content)
    assert list(content) == [1.1, 2.2, 3.3]
    assert [numpy.asarray(x).tolist() for x in array] == [[1.1, 2.2], [], [3.3]]

def test_refcount():
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]))
    offsets = awkward1.layout.Index32(numpy.array([0, 2, 2, 3], "i4"))
    array = awkward1.layout.ListOffsetArray32(offsets, content)

    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)

    iter1 = iter(content)
    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)
    x1 = next(iter1)
    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)

    iter2 = iter(array)
    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)
    x2 = next(iter2)
    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)

    del iter1
    del x1
    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)

    del iter2
    del x2
    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)

def test_numba_numpyarray():
    array = awkward1.layout.NumpyArray(numpy.arange(12))
    @numba.njit
    def f1(q):
        out = 0
        for x in q:
            out += x
        return out
    assert f1(array) == 66

    array = awkward1.layout.NumpyArray(numpy.arange(12).reshape(3, 4))
    @numba.njit
    def f2(q):
        out = 0
        for x in q:
            for y in x:
                out += y
        return out
    assert f2(array) == 66

def test_numba_listoffsetarray():
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]))
    offsets = awkward1.layout.Index32(numpy.array([0, 2, 2, 3], "i4"))
    array = awkward1.layout.ListOffsetArray32(offsets, content)

    @numba.njit
    def f1(q):
        out = 0.0
        for x in q:
            for y in x:
                out += y
        return out
    assert f1(array) == 6.6

    @numba.njit
    def f2(q):
        for x in q:
            return x
    assert numpy.asarray(f2(array)).tolist() == [1.1, 2.2]

def test_tolist():
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]))
    offsets = awkward1.layout.Index32(numpy.array([0, 2, 2, 3], "i4"))
    array = awkward1.layout.ListOffsetArray32(offsets, content)
    assert awkward1.tolist(array) == [[1.1, 2.2], [], [3.3]]
