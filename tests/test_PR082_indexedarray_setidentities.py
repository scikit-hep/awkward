# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_error():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.Index64(numpy.array([0, 2, 4, 6, 8, 10, 12, 14], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray64(index, content)

    with pytest.raises(ValueError) as err:
        indexedarray.setidentities()
    assert str(err.value) == "in IndexedArray64 attempting to get 10, max(index) > len(content)"

def test_passthrough_32():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.Index32(numpy.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=numpy.int32))
    indexedarray = awkward1.layout.IndexedArray32(index, content)

    assert awkward1.tolist(indexedarray) == [0.0, 2.2, 4.4, 6.6, 8.8, 9.9, 7.7, 5.5]
    indexedarray.setidentities()
    assert numpy.asarray(indexedarray.identities).tolist() == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert numpy.asarray(indexedarray.content.identities).tolist() == [[0], [-1], [1], [-1], [2], [7], [3], [6], [4], [5]]
    assert isinstance(indexedarray.content.identities, awkward1.layout.Identities32)

def test_passthrough_U32():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.IndexU32(numpy.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=numpy.uint32))
    indexedarray = awkward1.layout.IndexedArrayU32(index, content)

    assert awkward1.tolist(indexedarray) == [0.0, 2.2, 4.4, 6.6, 8.8, 9.9, 7.7, 5.5]
    indexedarray.setidentities()
    assert numpy.asarray(indexedarray.identities).tolist() == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert numpy.asarray(indexedarray.content.identities).tolist() == [[0], [-1], [1], [-1], [2], [7], [3], [6], [4], [5]]
    assert isinstance(indexedarray.content.identities, awkward1.layout.Identities64)

def test_passthrough_64():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.Index64(numpy.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray64(index, content)

    assert awkward1.tolist(indexedarray) == [0.0, 2.2, 4.4, 6.6, 8.8, 9.9, 7.7, 5.5]
    indexedarray.setidentities()
    assert numpy.asarray(indexedarray.identities).tolist() == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert numpy.asarray(indexedarray.content.identities).tolist() == [[0], [-1], [1], [-1], [2], [7], [3], [6], [4], [5]]
    assert isinstance(indexedarray.content.identities, awkward1.layout.Identities64)

def test_dontpass_32():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.Index32(numpy.array([0, 2, 4, 6, 8, 6, 4, 2, 0], dtype=numpy.int32))
    indexedarray = awkward1.layout.IndexedArray32(index, content)

    assert awkward1.tolist(indexedarray) == [0.0, 2.2, 4.4, 6.6, 8.8, 6.6, 4.4, 2.2, 0.0]
    indexedarray.setidentities()
    assert numpy.asarray(indexedarray.identities).tolist() == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    assert indexedarray.content.identities is None

def test_dontpass_U32():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.IndexU32(numpy.array([0, 2, 4, 6, 8, 6, 4, 2, 0], dtype=numpy.uint32))
    indexedarray = awkward1.layout.IndexedArrayU32(index, content)

    assert awkward1.tolist(indexedarray) == [0.0, 2.2, 4.4, 6.6, 8.8, 6.6, 4.4, 2.2, 0.0]
    indexedarray.setidentities()
    assert numpy.asarray(indexedarray.identities).tolist() == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    assert indexedarray.content.identities is None

def test_dontpass_64():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.Index64(numpy.array([0, 2, 4, 6, 8, 6, 4, 2, 0], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray64(index, content)

    assert awkward1.tolist(indexedarray) == [0.0, 2.2, 4.4, 6.6, 8.8, 6.6, 4.4, 2.2, 0.0]
    indexedarray.setidentities()
    assert numpy.asarray(indexedarray.identities).tolist() == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    assert indexedarray.content.identities is None

numba = pytest.importorskip("numba")

content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
index1 = awkward1.layout.Index64(numpy.array([2, 3, 3, 0, 4, 8], dtype=numpy.int64))
indexedarray1 = awkward1.layout.IndexedArray64(index1, content)
index2 = awkward1.layout.Index64(numpy.array([2, 3, 3, -1, -1, 8], dtype=numpy.int64))
indexedarray2 = awkward1.layout.IndexedOptionArray64(index2, content)
offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6], dtype=numpy.int64))
listoffsetarray1 = awkward1.layout.ListOffsetArray64(offsets1, indexedarray1)
listoffsetarray2 = awkward1.layout.ListOffsetArray64(offsets1, indexedarray2)
offsets3 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10], dtype=numpy.int64))
listoffsetarray3 = awkward1.layout.ListOffsetArray64(offsets3, content)
index3 = awkward1.layout.Index64(numpy.array([2, 0, 1, 3, 3, 4], dtype=numpy.int64))
indexedarray3 = awkward1.layout.IndexedArray64(index3, listoffsetarray3)
index4 = awkward1.layout.Index64(numpy.array([2, -1, -1, 3, 3, 4], dtype=numpy.int64))
indexedarray4 = awkward1.layout.IndexedOptionArray64(index4, listoffsetarray3)

def test_boxing():
    @numba.njit
    def f1(q):
        return 3.14

    f1(indexedarray1)
    f1(indexedarray2)
    f1(indexedarray3)
    f1(indexedarray4)

    @numba.njit
    def f2(q):
        return q

    assert awkward1.tolist(f2(indexedarray1)) == [2.2, 3.3, 3.3, 0.0, 4.4, 8.8]
    assert awkward1.tolist(f2(indexedarray2)) == [2.2, 3.3, 3.3, None, None, 8.8]
    assert awkward1.tolist(f2(indexedarray3)) == [[3.3, 4.4], [0.0, 1.1, 2.2], [], [5.5], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.tolist(f2(indexedarray4)) == [[3.3, 4.4], None, None, [5.5], [5.5], [6.6, 7.7, 8.8, 9.9]]

def test_len():
    @numba.njit
    def f1(q):
        return len(q)

    assert f1(indexedarray1) == 6
    assert f1(indexedarray2) == 6
    assert f1(indexedarray3) == 6
    assert f1(indexedarray4) == 6

def test_getitem_int():
    @numba.njit
    def f1(q, i):
        return q[i]

    assert f1(indexedarray1, 2) == 3.3
    assert f1(indexedarray1, 3) == 0.0
    assert f1(indexedarray2, 2) == 3.3
    assert f1(indexedarray2, 3) == None
    assert awkward1.tolist(f1(indexedarray3, 1)) == [0.0, 1.1, 2.2]
    assert awkward1.tolist(f1(indexedarray4, 1)) == None
    assert awkward1.tolist(f1(indexedarray4, 3)) == [5.5]

def test_iter():
    @numba.njit
    def f1(q):
        total = 0.0
        for x in q:
            total += x
        return total

    assert f1(indexedarray1) == 2.2 + 3.3 + 3.3 + 0.0 + 4.4 + 8.8

    @numba.njit
    def f1(q):
        total = 0.0
        for x in q:
            if x is not None:
                total += x
        return total

    assert f1(indexedarray2) == 2.2 + 3.3 + 3.3 + 8.8

    @numba.njit
    def f1(q):
        total = 0.0
        for x in q:
            for y in x:
                total += y
        return total

    assert f1(indexedarray3) == 3.3 + 4.4 + 0.0 + 1.1 + 2.2 + 5.5 + 5.5 + 6.6 + 7.7 + 8.8 + 9.9

    @numba.njit
    def f1(q):
        total = 0.0
        for x in q:
            if x is not None:
                for y in x:
                    total += y
        return total

    assert f1(indexedarray4) == 3.3 + 4.4 + 5.5 + 5.5 + 6.6 + 7.7 + 8.8 + 9.9

def test_getitem_range():
    @numba.njit
    def f1(q):
        return q[-4:]

    assert awkward1.tolist(f1(indexedarray1)) == [3.3, 0.0, 4.4, 8.8]
    assert awkward1.tolist(f1(indexedarray2)) == [3.3, None, None, 8.8]
    assert awkward1.tolist(f1(indexedarray3)) == [[], [5.5], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.tolist(f1(indexedarray4)) == [None, [5.5], [5.5], [6.6, 7.7, 8.8, 9.9]]

def test_carry():
    @numba.njit
    def f1(q):
        return q[-4:,]

    assert awkward1.tolist(f1(indexedarray1)) == [3.3, 0.0, 4.4, 8.8]
    assert awkward1.tolist(f1(indexedarray2)) == [3.3, None, None, 8.8]
    assert awkward1.tolist(f1(indexedarray3)) == [[], [5.5], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.tolist(f1(indexedarray4)) == [None, [5.5], [5.5], [6.6, 7.7, 8.8, 9.9]]

def test_getitem_next():
    @numba.njit
    def f1(q):
        return q[-4:, :]

    assert awkward1.tolist(f1(indexedarray3)) == [[], [5.5], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.tolist(f1(indexedarray4)) == [None, [5.5], [5.5], [6.6, 7.7, 8.8, 9.9]]

content2 = awkward1.Array([{"x": 0, "y": 0.0}, {"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}, {"x": 4, "y": 4.4}]).layout
index5 = awkward1.layout.Index64(numpy.array([3, 3, 0, 2, 1, 4, 0], dtype=numpy.int64))
indexedarray5 = awkward1.layout.IndexedArray64(index5, content2)
index6 = awkward1.layout.Index64(numpy.array([3, 3, -1, 2, -1, 4, 0], dtype=numpy.int64))
indexedarray6 = awkward1.layout.IndexedOptionArray64(index6, content2)

def test_getitem_str():
    assert awkward1.tolist(indexedarray5) == [{"x": 3, "y": 3.3}, {"x": 3, "y": 3.3}, {"x": 0, "y": 0.0}, {"x": 2, "y": 2.2}, {"x": 1, "y": 1.1}, {"x": 4, "y": 4.4}, {"x": 0, "y": 0.0}]
    assert awkward1.tolist(indexedarray6) == [{"x": 3, "y": 3.3}, {"x": 3, "y": 3.3}, None, {"x": 2, "y": 2.2}, None, {"x": 4, "y": 4.4}, {"x": 0, "y": 0.0}]

    @numba.njit
    def f1(q):
        return q["y"]

    assert awkward1.tolist(f1(indexedarray5)) == [3.3, 3.3, 0.0, 2.2, 1.1, 4.4, 0.0]
    assert awkward1.tolist(f1(indexedarray6)) == [3.3, 3.3, None, 2.2, None, 4.4, 0.0]

    @numba.njit
    def f2(q):
        return q["y", :]

    assert awkward1.tolist(f2(indexedarray5)) == [3.3, 3.3, 0.0, 2.2, 1.1, 4.4, 0.0]
    assert awkward1.tolist(f2(indexedarray6)) == [3.3, 3.3, None, 2.2, None, 4.4, 0.0]

    @numba.njit
    def f3(q):
        return q[:, "y"]

    assert awkward1.tolist(f2(indexedarray5)) == [3.3, 3.3, 0.0, 2.2, 1.1, 4.4, 0.0]
    assert awkward1.tolist(f2(indexedarray6)) == [3.3, 3.3, None, 2.2, None, 4.4, 0.0]
