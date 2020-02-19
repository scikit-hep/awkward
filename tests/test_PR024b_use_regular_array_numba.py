# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import itertools

import pytest
import numpy

import awkward1

pytest.skip("Disabling Numba until the rewrite is done.", allow_module_level=True)

numba = pytest.importorskip("numba")

def test_empty_array_slice_numba():
    # inspired by PR015::test_deep_numpy
    content = awkward1.layout.NumpyArray(numpy.array([[0.0, 1.1], [2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]]))
    listarray = awkward1.layout.ListArray64(awkward1.layout.Index64(numpy.array([0, 3, 3])), awkward1.layout.Index64(numpy.array([3, 3, 5])), content)

    @numba.njit
    def f1(q, i, j):
        return q[2, i, j]

    assert awkward1.tolist(f1(listarray, numpy.array([1], dtype=int), numpy.array([], dtype=int))) == []
    assert awkward1.tolist(f1(listarray, numpy.array([], dtype=int), numpy.array([], dtype=int))) == []
