# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools

import pytest
import numpy
numba = pytest.importorskip("numba")

import awkward1

py27 = (sys.version_info[0] < 3)

def test_getitem_array_64():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    listarray = awkward1.layout.ListArray64(awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6])), awkward1.layout.Index64(numpy.array([3, 3, 5, 6, 10])), content)

    @numba.njit
    def f1(q):
        return q[[2, 0, 0, 1]]

    assert awkward1.tolist(f1(listarray)) == [[3.3, 4.4], [0.0, 1.1, 2.2], [0.0, 1.1, 2.2], []]

    @numba.njit
    def f2(q):
        return q[[2, 0, 0, -1], 1]

    assert awkward1.tolist(f2(listarray)) == [4.4, 1.1, 1.1, 7.7]

    @numba.njit
    def f3(q):
        return q[[2, 0, 0, -1], [-1, -1, 0, 0]]

    assert awkward1.tolist(f3(listarray)) == [4.4, 2.2, 0.0, 6.6]

    listarray.setid()
    assert numpy.asarray(f1(listarray).id).tolist() == [[2], [0], [0], [1]]
    assert numpy.asarray(f2(listarray).id).tolist() == [[2, 1], [0, 1], [0, 1], [4, 1]]
    assert numpy.asarray(f3(listarray).id).tolist() == [[2, 1], [0, 2], [0, 0], [4, 0]]

def test_getitem_array_32():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    listarray = awkward1.layout.ListArray32(awkward1.layout.Index32(numpy.array([0, 3, 3, 5, 6], numpy.int32)), awkward1.layout.Index32(numpy.array([3, 3, 5, 6, 10], numpy.int32)), content)

    @numba.njit
    def f1(q):
        return q[[2, 0, 0, 1]]

    assert awkward1.tolist(f1(listarray)) == [[3.3, 4.4], [0.0, 1.1, 2.2], [0.0, 1.1, 2.2], []]

    @numba.njit
    def f2(q):
        return q[[2, 0, 0, -1], 1]

    assert awkward1.tolist(f2(listarray)) == [4.4, 1.1, 1.1, 7.7]

    @numba.njit
    def f3(q):
        return q[[2, 0, 0, -1], [-1, -1, 0, 0]]

    assert awkward1.tolist(f3(listarray)) == [4.4, 2.2, 0.0, 6.6]

    listarray.setid()
    assert numpy.asarray(f1(listarray).id).tolist() == [[2], [0], [0], [1]]
    assert numpy.asarray(f2(listarray).id).tolist() == [[2, 1], [0, 1], [0, 1], [4, 1]]
    assert numpy.asarray(f3(listarray).id).tolist() == [[2, 1], [0, 2], [0, 0], [4, 0]]

def test_deep_numpy():
    content = awkward1.layout.NumpyArray(numpy.array([[0.0, 1.1], [2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]]))
    listarray = awkward1.layout.ListArray64(awkward1.layout.Index64(numpy.array([0, 3, 3])), awkward1.layout.Index64(numpy.array([3, 3, 5])), content)
    assert awkward1.tolist(listarray) == [[[0.0, 1.1], [2.2, 3.3], [4.4, 5.5]], [], [[6.6, 7.7], [8.8, 9.9]]]

    @numba.njit
    def f1(q):
        return q[[2, 0, 0, 1]]

    assert awkward1.tolist(f1(content)) == [[4.4, 5.5], [0.0, 1.1], [0.0, 1.1], [2.2, 3.3]]

content = awkward1.layout.NumpyArray(numpy.arange(2*3*5*7).reshape(-1, 7))
offsetsA = numpy.arange(0, 2*3*5 + 5, 5)
offsetsB = numpy.arange(0, 2*3 + 3, 3)
startsA, stopsA = offsetsA[:-1], offsetsA[1:]
startsB, stopsB = offsetsB[:-1], offsetsB[1:]

listoffsetarrayA64 = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(offsetsA), content)
listoffsetarrayA32 = awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(offsetsA), content)
listarrayA64 = awkward1.layout.ListArray64(awkward1.layout.Index64(startsA), awkward1.layout.Index64(stopsA), content)
listarrayA32 = awkward1.layout.ListArray32(awkward1.layout.Index32(startsA), awkward1.layout.Index32(stopsA), content)
modelA = numpy.arange(2*3*5*7).reshape(2*3, 5, 7)

listoffsetarrayB64 = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(offsetsB), listoffsetarrayA64)
listoffsetarrayB32 = awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(offsetsB), listoffsetarrayA64)
listarrayB64 = awkward1.layout.ListArray64(awkward1.layout.Index64(startsB), awkward1.layout.Index64(stopsB), listarrayA64)
listarrayB32 = awkward1.layout.ListArray32(awkward1.layout.Index32(startsB), awkward1.layout.Index32(stopsB), listarrayA64)
modelB = numpy.arange(2*3*5*7).reshape(2, 3, 5, 7)
