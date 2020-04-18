# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

numba = pytest.importorskip("numba")

awkward1_numba_arrayview = pytest.importorskip("awkward1._connect._numba.arrayview")

def test_view():
    aslist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    asarray = awkward1.repartition(awkward1.Array(aslist), 3)
    asview = awkward1_numba_arrayview.ArrayView.fromarray(asarray)

    for start in range(10):
        for stop in range(start, 10):
            asview.start = start
            asview.stop = stop
            assert awkward1.to_list(asview.toarray()) == aslist[start:stop]

    asarray = awkward1.repartition(awkward1.Array(aslist), [3, 2, 0, 1, 4])
    asview = awkward1_numba_arrayview.ArrayView.fromarray(asarray)

    for start in range(10):
        for stop in range(start, 10):
            asview.start = start
            asview.stop = stop
            assert awkward1.to_list(asview.toarray()) == aslist[start:stop]

    aslist = [[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]]
    asarray = awkward1.repartition(awkward1.Array(aslist), 3)
    asview = awkward1_numba_arrayview.ArrayView.fromarray(asarray)

    for start in range(5):
        for stop in range(start, 5):
            asview.start = start
            asview.stop = stop
            assert awkward1.to_list(asview.toarray()) == aslist[start:stop]

def test_numba():
    array = awkward1.repartition(awkward1.Array([[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]]), 3)

    # @numba.njit
    # def f1(x):
    #     return 3.14

    # f1(array)

    @numba.njit
    def f2(x):
        return x

    print(f2(array))
    raise Exception
