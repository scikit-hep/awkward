# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools

import pytest
import numpy

numba = pytest.importorskip("numba")

import awkward1

content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content2)
recordarray = awkward1.layout.RecordArray({"one": content1, "two": listoffsetarray})

def test_boxing():
    @numba.njit
    def f1(q):
        return 3.14

    assert f1(recordarray) == 3.14
    assert f1(recordarray[2]) == 3.14

    @numba.njit
    def f2(q):
        return q

    assert awkward1.tolist(f2(recordarray)) == [{"one": 1, "two": [1.1, 2.2, 3.3]}, {"one": 2, "two": []}, {"one": 3, "two": [4.4, 5.5]}, {"one": 4, "two": [6.6]}, {"one": 5, "two": [7.7, 8.8, 9.9]}]
    assert awkward1.tolist(f2(recordarray[2])) == {"one": 3, "two": [4.4, 5.5]}

def test_len():
    @numba.njit
    def f1(q):
        return len(q)

    assert f1(recordarray) == 5
    with pytest.raises(numba.TypingError):
        f1(recordarray[2])

def test_getitem_int():
    @numba.njit
    def f1(q):
        return q[2]

    assert awkward1.tolist(f1(recordarray)) == {"one": 3, "two": [4.4, 5.5]}
    with pytest.raises(numba.TypingError):
        f1(recordarray[2])

def test_getitem_iter():
    @numba.njit
    def f1(q):
        out = 0
        for x in q:
            out += 1
        return out

    assert f1(recordarray) == 5
    with pytest.raises(numba.TypingError):
        f1(recordarray[2])

def test_getitem_range():
    @numba.njit
    def f1(q):
        return q[1:4]

    assert awkward1.tolist(f1(recordarray)) == [{"one": 2, "two": []}, {"one": 3, "two": [4.4, 5.5]}, {"one": 4, "two": [6.6]}]
    with pytest.raises(numba.TypingError):
        f1(recordarray[2])

def test_getitem_str():
    outer_starts = numpy.array([0, 3, 3], dtype=numpy.int64)
    outer_stops = numpy.array([3, 3, 5], dtype=numpy.int64)
    outer_offsets = numpy.array([0, 3, 3, 5], dtype=numpy.int64)
    outer_listarray = awkward1.layout.ListArray64(awkward1.layout.Index64(outer_starts), awkward1.layout.Index64(outer_stops), recordarray)
    outer_listoffsetarray = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(outer_offsets), recordarray)
    outer_regulararray = awkward1.layout.RegularArray(recordarray, 2)

    @numba.njit
    def f1(q):
        return q["one"]

    assert awkward1.tolist(f1(recordarray)) == [1, 2, 3, 4, 5]

    assert sys.getrefcount(outer_starts), sys.getrefcount(outer_stops) == (3, 3)
    assert awkward1.tolist(f1(outer_listarray)) == [[1, 2, 3], [], [4, 5]]
    assert sys.getrefcount(outer_starts), sys.getrefcount(outer_stops) == (3, 3)

    assert sys.getrefcount(outer_offsets) == 3
    assert awkward1.tolist(f1(outer_listoffsetarray)) == [[1, 2, 3], [], [4, 5]]
    assert sys.getrefcount(outer_offsets) == 3

    assert awkward1.tolist(f1(outer_regulararray)) == [[1, 2], [3, 4]]

    @numba.njit
    def f2(q):
        return q["two"]

    assert awkward1.tolist(f1(recordarray[2])) == 3
    assert awkward1.tolist(f2(recordarray[2])) == [4.4, 5.5]

def test_getitem_tuple():
    content3 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    regulararray = awkward1.layout.RegularArray(content3, 2)
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 4, 5, 8, 9]))
    listoffsetarray2 = awkward1.layout.ListOffsetArray64(offsets2, content2)
    recordarray2 = awkward1.layout.RecordArray({"one": regulararray, "two": listoffsetarray2})

    assert awkward1.tolist(recordarray2) == [{"one": [1, 2], "two": [1.1, 2.2, 3.3]}, {"one": [3, 4], "two": [4.4]}, {"one": [5, 6], "two": [5.5]}, {"one": [7, 8], "two": [6.6, 7.7, 8.8]}, {"one": [9, 10], "two": [9.9]}]

    @numba.njit
    def f1(q):
        return q[:, -1]

    assert awkward1.tolist(f1(recordarray2)) == [{"one": 2, "two": 3.3}, {"one": 4, "two": 4.4}, {"one": 6, "two": 5.5}, {"one": 8, "two": 8.8}, {"one": 10, "two": 9.9}]

    @numba.njit
    def f2(q):
        return q[:, -2:]

    assert awkward1.tolist(f2(recordarray2)) == [{"one": [1, 2], "two": [2.2, 3.3]}, {"one": [3, 4], "two": [4.4]}, {"one": [5, 6], "two": [5.5]}, {"one": [7, 8], "two": [7.7, 8.8]}, {"one": [9, 10], "two": [9.9]}]

    @numba.njit
    def f3(q):
        return q[2:, "two"]

    assert awkward1.tolist(f3(recordarray2)) == [[5.5], [6.6, 7.7, 8.8], [9.9]]

    @numba.njit
    def f4(q):
        return q["two", 1:-1]

    assert awkward1.tolist(f4(recordarray2)) == [[4.4], [5.5], [6.6, 7.7, 8.8]]

def test_fillablearray_tuple():
    fillablearray = awkward1.layout.FillableArray()

    @numba.njit
    def f1(q):
        q.begintuple(3)
        q.index(0); q.boolean(True)
        q.index(1); q.integer(1)
        q.index(2); q.real(1.1)
        q.endtuple()

        q.begintuple(3)
        q.index(0); q.boolean(False)
        q.index(1); q.integer(1)
        q.index(2); q.real(1.1)
        q.endtuple()

        q.begintuple(3)
        q.index(0); q.boolean(True)
        q.index(1); q.integer(1)
        q.index(2); q.real(1.1)
        q.endtuple()

        return q

    fillablearray2 = f1(fillablearray)

    assert awkward1.tolist(fillablearray.snapshot()) == [(True, 1, 1.1), (False, 1, 1.1), (True, 1, 1.1)]
    assert awkward1.tolist(fillablearray2.snapshot()) == [(True, 1, 1.1), (False, 1, 1.1), (True, 1, 1.1)]

def test_fillablearray_record_1():
    fillablearray = awkward1.layout.FillableArray()

    @numba.njit
    def f1(q):
        q.beginrecord()
        q.field("one"); q.boolean(True)
        q.field("two"); q.integer(1)
        q.field("three"); q.real(1.1)
        q.endrecord()

        q.beginrecord()
        q.field("one"); q.boolean(False)
        q.field("two"); q.integer(2)
        q.field("three"); q.real(2.2)
        q.endrecord()

        q.beginrecord()
        q.field("one"); q.boolean(True)
        q.field("two"); q.integer(3)
        q.field("three"); q.real(3.3)
        q.endrecord()

        return q

    fillablearray2 = f1(fillablearray)

    assert awkward1.tolist(fillablearray.snapshot()) == [{'one': True, 'two': 1, 'three': 1.1}, {'one': False, 'two': 2, 'three': 2.2}, {'one': True, 'two': 3, 'three': 3.3}]
    assert awkward1.tolist(fillablearray2.snapshot()) == [{'one': True, 'two': 1, 'three': 1.1}, {'one': False, 'two': 2, 'three': 2.2}, {'one': True, 'two': 3, 'three': 3.3}]

def test_fillablearray_record_2():
    fillablearray = awkward1.layout.FillableArray()

    @numba.njit
    def f1(q):
        q.beginrecord("wowie")
        q.field("one"); q.boolean(True)
        q.field("two"); q.integer(1)
        q.field("three"); q.real(1.1)
        q.endrecord()

        q.beginrecord("wowie")
        q.field("one"); q.boolean(False)
        q.field("two"); q.integer(2)
        q.field("three"); q.real(2.2)
        q.endrecord()

        q.beginrecord("wowie")
        q.field("one"); q.boolean(True)
        q.field("two"); q.integer(3)
        q.field("three"); q.real(3.3)
        q.endrecord()

        return q

    fillablearray2 = f1(fillablearray)

    assert awkward1.tolist(fillablearray.snapshot()) == [{'one': True, 'two': 1, 'three': 1.1}, {'one': False, 'two': 2, 'three': 2.2}, {'one': True, 'two': 3, 'three': 3.3}]
    assert awkward1.tolist(fillablearray2.snapshot()) == [{'one': True, 'two': 1, 'three': 1.1}, {'one': False, 'two': 2, 'three': 2.2}, {'one': True, 'two': 3, 'three': 3.3}]
