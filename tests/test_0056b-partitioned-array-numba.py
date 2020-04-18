# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

numba = pytest.importorskip("numba")

awkward1_numba_arrayview = pytest.importorskip("awkward1._connect._numba.arrayview")

def test_view():
    array = awkward1.repartition(awkward1.Array([[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]]), 3)

    print(awkward1_numba_arrayview.ArrayView.fromarray(array).toarray())
    raise Exception


# def test_numba():
#     array = awkward1.repartition(awkward1.Array([[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]]), 3)

#     @numba.njit
#     def f1(x):
#         return 3.14

#     f1(array)
