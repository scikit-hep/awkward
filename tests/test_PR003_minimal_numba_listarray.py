# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

pytest.skip("Disabling Numba until the rewrite is done.", allow_module_level=True)

numba = pytest.importorskip("numba")

import awkward1

py27 = 2 if sys.version_info[0] < 3 else 1

def test_numpyarray_boxing():
    a = numpy.arange(10)
    wrapped = awkward1.layout.NumpyArray(a)
    assert (sys.getrefcount(a), sys.getrefcount(wrapped)) == (3, 2)

    @numba.njit
    def f1(q):
        pass
    f1(wrapped)

    @numba.njit
    def f2(q):
        return q
    out = f2(wrapped)

    assert (sys.getrefcount(a), sys.getrefcount(wrapped)) == (3, 2 + 1*py27)
    assert numpy.asarray(out).tolist() == list(range(10))
    del out
    assert (sys.getrefcount(a), sys.getrefcount(wrapped)) == (3, 2)

def test_numpyarray_refcount1():
    a = numpy.arange(10)
    wrapped = awkward1.layout.NumpyArray(a)
    assert (sys.getrefcount(a), sys.getrefcount(wrapped)) == (3, 2)
    @numba.njit
    def f1(q):
        return q[1:9][1:8][1:7]
    out = f1(wrapped)
    assert (sys.getrefcount(a), sys.getrefcount(wrapped)) == (3, 2 + 1*py27)
    assert numpy.asarray(out).tolist() == list(range(10)[1:9][1:8][1:7])
    del out
    assert (sys.getrefcount(a), sys.getrefcount(wrapped)) == (3, 2)

def test_numpyarray_refcount2():
    a = numpy.arange(10)
    wrapped = awkward1.layout.NumpyArray(a)
    assert (sys.getrefcount(a), sys.getrefcount(wrapped)) == (3, 2)
    @numba.njit
    def f1(q):
        return q[1:9], q[2:8], q[3:7]
    out = f1(wrapped)
    assert (sys.getrefcount(a), sys.getrefcount(wrapped)) == (3, 2 + 1*py27)
    assert numpy.asarray(out[-1]).tolist() == list(range(3, 7))
    del out
    assert (sys.getrefcount(a), sys.getrefcount(wrapped)) == (3, 2)

def test_numpyarray_len():
    a = awkward1.layout.NumpyArray(numpy.arange(10))
    @numba.njit
    def f1(q):
        return len(q)
    assert f1(a) == 10

def test_numpyarray_getitem_int():
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

    @numba.njit
    def f4(q, i):
        return q[i]
    assert numpy.asarray(f4(a, 1)).tolist() == [4, 5, 6, 7]

def test_numpyarray_getitem_slice():
    a = awkward1.layout.NumpyArray(numpy.arange(12).reshape(3, 4))

    @numba.njit
    def f1(q):
        return q[1:]
    out = f1(a)
    assert isinstance(out, awkward1.layout.NumpyArray)
    assert numpy.asarray(out).tolist() == [[4, 5, 6, 7], [8, 9, 10, 11]]

    @numba.njit
    def f2(q, i):
        return q[i:]
    out = f2(a, 1)
    assert isinstance(out, awkward1.layout.NumpyArray)
    assert numpy.asarray(out).tolist() == [[4, 5, 6, 7], [8, 9, 10, 11]]

    @numba.njit
    def f3(q, i):
        return q[i]
    out = f3(a, slice(1, None))
    assert isinstance(out, awkward1.layout.NumpyArray)
    assert numpy.asarray(out).tolist() == [[4, 5, 6, 7], [8, 9, 10, 11]]

def test_listoffsetarray_boxing():
    offsets = awkward1.layout.Index32(numpy.array([0, 2, 2, 3], "i4"))
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]))
    array = awkward1.layout.ListOffsetArray32(offsets, content)
    assert (sys.getrefcount(offsets), sys.getrefcount(content), sys.getrefcount(array)) == (2, 2, 2)

    @numba.njit
    def f1(q):
        return q
    out = f1(array)
    assert (sys.getrefcount(offsets), sys.getrefcount(content), sys.getrefcount(array)) == (2, 2, 2)
    assert numpy.asarray(out.offsets).tolist() == [0, 2, 2, 3]
    assert numpy.asarray(out.content).tolist() == [1.1, 2.2, 3.3]

    array2 = awkward1.layout.ListOffsetArray32(offsets, array)
    out2 = f1(array2)
    assert (sys.getrefcount(offsets), sys.getrefcount(content), sys.getrefcount(array)) == (2, 2, 2)
    assert numpy.asarray(out2.offsets).tolist() == [0, 2, 2, 3]
    assert numpy.asarray(out2.content.offsets).tolist() == [0, 2, 2, 3]
    assert numpy.asarray(out2.content.content).tolist() == [1.1, 2.2, 3.3]

def test_listoffsetarray_len():
    offsets = awkward1.layout.Index32(numpy.array([0, 2, 2, 3], "i4"))
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]))
    array = awkward1.layout.ListOffsetArray32(offsets, content)
    @numba.njit
    def f1(q):
        return len(q)
    assert f1(array) == 3

def test_listoffsetarray_getitem_int():
    offsets = awkward1.layout.Index32(numpy.array([0, 2, 2, 3], "i4"))
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]))
    array = awkward1.layout.ListOffsetArray32(offsets, content)
    @numba.njit
    def f1(q):
        return q[0]
    assert numpy.asarray(f1(array)).tolist() == [1.1, 2.2]
    @numba.njit
    def f2(q, i):
        return q[i]
    assert numpy.asarray(f2(array, 0)).tolist() == [1.1, 2.2]
    assert numpy.asarray(f2(array, 1)).tolist() == []
    assert numpy.asarray(f2(array, 2)).tolist() == [3.3]

def test_listoffsetarray_getitem_slice():
    offsets = awkward1.layout.Index32(numpy.array([0, 2, 2, 3], "i4"))
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]))
    array = awkward1.layout.ListOffsetArray32(offsets, content)
    @numba.njit
    def f1(q):
        return q[1:][1]
    assert numpy.asarray(f1(array)).tolist() == [3.3]
    @numba.njit
    def f2(q, i):
        return q[i:]
    out = f2(array, 1)
    assert numpy.asarray(out[0]).tolist() == []
    assert numpy.asarray(out[1]).tolist() == [3.3]
    @numba.njit
    def f3(q, i):
        return q[i]
    out = f3(array, slice(1, None))
    assert numpy.asarray(out[0]).tolist() == []
    assert numpy.asarray(out[1]).tolist() == [3.3]
