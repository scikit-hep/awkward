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
    identity = awkward1.layout.Identities32(0, [], array)
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

def test_width():
    identity = awkward1.layout.Identities32(0, [], numpy.arange(40, dtype="i4").reshape(-1, 4))
    @numba.njit
    def f1(q):
        return q.width
    assert f1(identity) == 4

def test_numpyarray():
    i1 = numpy.arange(3, dtype="i4").reshape(-1, 1)
    i2 = awkward1.layout.Identities32(0, [], i1)
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]), id=i2)

    assert (sys.getrefcount(i1), sys.getrefcount(i2), sys.getrefcount(content)) == (3, 2, 2)

    @numba.njit
    def f1(q):
        return q

    tmp = f1(content)
    assert (sys.getrefcount(i1), sys.getrefcount(i2), sys.getrefcount(content)) == (3, 2, 2 + 1*py27)

    tmp2 = tmp.id
    assert (sys.getrefcount(i1), sys.getrefcount(i2), sys.getrefcount(content)) == (3, 2, 2 + 1*py27)

    del tmp
    assert (sys.getrefcount(i1), sys.getrefcount(i2), sys.getrefcount(content)) == (3, 2, 2)

def test_listoffsetarray():
    i1 = numpy.arange(3, dtype="i4").reshape(-1, 1)
    i2 = awkward1.layout.Identities32(0, [], i1)
    offsets = awkward1.layout.Index32(numpy.array([0, 2, 2, 3], "i4"))
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]))
    array = awkward1.layout.ListOffsetArray32(offsets, content, id=i2)

    assert (sys.getrefcount(i1), sys.getrefcount(i2), sys.getrefcount(array)) == (3, 2, 2)

    @numba.njit
    def f1(q):
        return q

    tmp = f1(array)
    assert (sys.getrefcount(i1), sys.getrefcount(i2), sys.getrefcount(array)) == (3, 2, 2)

    tmp2 = tmp.id
    assert (sys.getrefcount(i1), sys.getrefcount(i2), sys.getrefcount(array)) == (3, 2, 2)

    del tmp
    assert (sys.getrefcount(i1), sys.getrefcount(i2), sys.getrefcount(array)) == (3, 2, 2)

def test_id_attribute():
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]))
    content.setid()
    assert numpy.asarray(content.id).tolist() == [[0], [1], [2]]

    @numba.njit
    def f1(q):
        return q.id

    assert numpy.asarray(f1(content)).tolist() == [[0], [1], [2]]

    offsets = awkward1.layout.Index32(numpy.array([0, 2, 2, 3], "i4"))
    array = awkward1.layout.ListOffsetArray32(offsets, content)
    array.setid()
    assert numpy.asarray(array.id).tolist() == [[0], [1], [2]]
    assert numpy.asarray(array.content.id).tolist() == [[0, 0], [0, 1], [2, 0]]

    @numba.njit
    def f2(q):
        return q.id

    assert numpy.asarray(f2(array)).tolist() == [[0], [1], [2]]

def test_other_attributes():
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]))
    offsets = awkward1.layout.Index32(numpy.array([0, 2, 2, 3], "i4"))
    array = awkward1.layout.ListOffsetArray32(offsets, content)

    @numba.njit
    def f1(q):
        return q.content

    assert numpy.asarray(f1(array)).tolist() == [1.1, 2.2, 3.3]

    @numba.njit
    def f2(q):
        return q.offsets

    assert numpy.asarray(f2(array)).tolist() == [0, 2, 2, 3]
