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

    assert awkward1.tolist(f2(recordarray)) == [{'one': 1, 'two': [1.1, 2.2, 3.3]}, {'one': 2, 'two': []}, {'one': 3, 'two': [4.4, 5.5]}, {'one': 4, 'two': [6.6]}, {'one': 5, 'two': [7.7, 8.8, 9.9]}]
    assert awkward1.tolist(f2(recordarray[2])) == {'one': 3, 'two': [4.4, 5.5]}

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

    assert awkward1.tolist(f1(recordarray)) == {'one': 3, 'two': [4.4, 5.5]}
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

def test_getitem_range():
    @numba.njit
    def f1(q):
        return q[1:4]

    assert awkward1.tolist(f1(recordarray)) == [{'one': 2, 'two': []}, {'one': 3, 'two': [4.4, 5.5]}, {'one': 4, 'two': [6.6]}]
