# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import itertools

import pytest
import numpy

import awkward1

content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]));
offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
regulararray = awkward1.layout.RegularArray(listoffsetarray, 2)
starts = awkward1.layout.Index64(numpy.array([0, 1]))
stops = awkward1.layout.Index64(numpy.array([2, 3]))
listarray = awkward1.layout.ListArray64(starts, stops, regulararray)

pytest.skip("Disabling Numba until the rewrite is done.", allow_module_level=True)

numba = pytest.importorskip("numba")

def test_numba():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]));
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
    regulararray = awkward1.layout.RegularArray(listoffsetarray, 2)
    regulararray_m1 = awkward1.layout.RegularArray(listoffsetarray[:-1], 2)
    regulararray_m2 = awkward1.layout.RegularArray(listoffsetarray[:-2], 2)
    starts = awkward1.layout.Index64(numpy.array([0, 1]))
    stops = awkward1.layout.Index64(numpy.array([2, 3]))
    listarray = awkward1.layout.ListArray64(starts, stops, regulararray)

    @numba.njit
    def f1(q):
        return 3.14

    assert sys.getrefcount(regulararray) == 2
    f1(regulararray)
    assert sys.getrefcount(regulararray) == 2

    @numba.njit
    def f2(q):
        return q

    assert sys.getrefcount(regulararray) == 2
    assert awkward1.tolist(f2(regulararray)) == awkward1.tolist(regulararray)
    assert sys.getrefcount(regulararray) == 2

    @numba.njit
    def f3(q):
        return len(q)

    assert f3(regulararray) == 3
    assert f3(regulararray_m1) == 2
    assert f3(regulararray_m2) == 2
    assert len(regulararray) == 3
    assert len(regulararray_m1) == 2
    assert len(regulararray_m2) == 2

    @numba.njit
    def f4(q):
        return q[1]

    assert awkward1.tolist(f4(regulararray)) == [[3.3, 4.4], [5.5]]

    @numba.njit
    def f5(q, i):
        return q[i]

    assert awkward1.tolist(f5(regulararray, 1)) == [[3.3, 4.4], [5.5]]

    @numba.njit
    def f6(q, i):
        return q[i]

    assert awkward1.tolist(f6(regulararray, slice(1, None))) == awkward1.tolist(regulararray[slice(1, None)]) == [[[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]
    assert awkward1.tolist(f6(regulararray, slice(None, -1))) == awkward1.tolist(regulararray[slice(None, -1)]) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]]]

    @numba.njit
    def f7(q, i):
        return q[(i,)]

    assert awkward1.tolist(f7(regulararray, 1)) == [[3.3, 4.4], [5.5]]

    @numba.njit
    def f8(q, i):
        return q[:, i]

    assert awkward1.tolist(f8(listarray, 1)) == awkward1.tolist(listarray[:, 1]) == [[[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]

    @numba.njit
    def f9(q):
        return q[:, 1, 0]

    assert awkward1.tolist(listarray[:, 1, 0]) == awkward1.tolist(f9(listarray)) == [[3.3, 4.4], [6.6, 7.7, 8.8, 9.9]]

    @numba.njit
    def f10(q, i):
        return q[:, 1, i]

    assert awkward1.tolist(f10(listarray, slice(None, 1))) == awkward1.tolist(listarray[:, 1, slice(None, 1)]) == [[[3.3, 4.4]], [[6.6, 7.7, 8.8, 9.9]]]
    assert awkward1.tolist(f10(listarray, slice(None, None))) == [[[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]
    assert awkward1.tolist(f10(listarray, slice(None, None, -1))) == [[[5.5], [3.3, 4.4]], [[], [6.6, 7.7, 8.8, 9.9]]]

    @numba.njit
    def f11(q, i):
        return q[[0, 1], 1, i]

    assert awkward1.tolist(f11(listarray, slice(None, 1))) == awkward1.tolist(listarray[[0, 1], 1, slice(None, 1)]) == [[[3.3, 4.4]], [[6.6, 7.7, 8.8, 9.9]]]

    @numba.njit
    def f12(q):
        return q[:, 1:, [1, 0, 1]]

    assert awkward1.tolist(f12(listarray)) == awkward1.tolist(listarray[:, 1:, [1, 0, 1]]) == [[[[5.5], [3.3, 4.4], [5.5]]], [[[], [6.6, 7.7, 8.8, 9.9], []]]]

    @numba.njit
    def f13(q):
        return q[:, [1], [1, 0, 1]]

    assert awkward1.tolist(f13(listarray)) == awkward1.tolist(listarray[:, [1], [1, 0, 1]]) == [[[5.5], [3.3, 4.4], [5.5]], [[], [6.6, 7.7, 8.8, 9.9], []]]

    @numba.njit
    def f14(q):
        return q.content

    assert awkward1.tolist(f14(regulararray)) == awkward1.tolist(regulararray.content)

    @numba.njit
    def f15(q):
        return q.size

    assert awkward1.tolist(f15(regulararray)) == awkward1.tolist(regulararray.size) == 2
