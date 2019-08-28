# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import gc

import pytest
import numpy
numba = pytest.importorskip("numba")

import awkward1

py27 = 2 if sys.version_info[0] < 3 else 1

def test_reminder():
    x1 = numpy.arange(10, dtype="i4")
    x2 = awkward1.layout.NumpyArray(x1)
    assert (sys.getrefcount(x1), sys.getrefcount(x2)) == (3, 2)

    @numba.njit
    def f1(q):
        pass
    f1(x2)
    assert (sys.getrefcount(x1), sys.getrefcount(x2)) == (3, 2)

    @numba.njit
    def f2(q):
        return q
    tmp = f2(x2)
    assert (sys.getrefcount(x1), sys.getrefcount(x2)) == (3, 2 + 1*py27)

    del tmp
    assert (sys.getrefcount(x1), sys.getrefcount(x2)) == (3, 2)

    del x2
    assert sys.getrefcount(x1) == 2

def test_refcount():
    array = numpy.arange(40, dtype="i4").reshape(-1, 4)
    identity = awkward1.layout.Identity(0, [], 1, 2, array)
    assert (sys.getrefcount(array), sys.getrefcount(identity)) == (3, 2)

    @numba.njit
    def f1(q):
        pass
    f1(identity)
    assert (sys.getrefcount(array), sys.getrefcount(identity)) == (3, 2)

    @numba.njit
    def f2(q):
        return q
    tmp = f2(identity)
    assert (sys.getrefcount(array), sys.getrefcount(identity)) == (3, 2 + 1*py27)

    del tmp
    assert (sys.getrefcount(array), sys.getrefcount(identity)) == (3, 2)

    tmp = f2(identity)
    assert (sys.getrefcount(array), sys.getrefcount(identity)) == (3, 2 + 1*py27)

    del array
    del identity
    del tmp

def test_keydepth():
    identity = awkward1.layout.Identity(0, [], 1, 2, numpy.arange(40, dtype="i4").reshape(-1, 4))
    @numba.njit
    def f1(q):
        return q.chunkdepth, q.indexdepth, q.keydepth
    assert f1(identity) == (1, 2, 4)
