# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools

import pytest
import numpy

import awkward1

def test_empty_array_slice():
    # inspired by PR021::test_getitem
    a = awkward1.fromjson("[[], [[], []], [[], [], []]]")
    assert awkward1.tolist(a[2, 1, numpy.array([], dtype=int)]) == []
    assert awkward1.tolist(a[2, numpy.array([1], dtype=int), numpy.array([], dtype=int)]) == []

    # inspired by PR015::test_deep_numpy
    content = awkward1.layout.NumpyArray(numpy.array([[0.0, 1.1], [2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]]))
    listarray = awkward1.layout.ListArray64(awkward1.layout.Index64(numpy.array([0, 3, 3])), awkward1.layout.Index64(numpy.array([3, 3, 5])), content)
    assert awkward1.tolist(listarray[[2, 0, 0, -1], [1, -1, 0, 0], [0, 1, 0, 1]]) == [8.8, 5.5, 0.0, 7.7]
    assert awkward1.tolist(listarray[2, 1, numpy.array([], dtype=int)]) == []
    assert awkward1.tolist(listarray[2, 1, []]) == []
    assert awkward1.tolist(listarray[2, [1], []]) == []
    assert awkward1.tolist(listarray[2, [], []]) == []

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

# Sequential:
#############
# TODO: all getitem arrays should handle non-flat SliceArray by wrapping in RegularArrays.
# TODO: all of the above should happen in Numba, too.

# Independent:
##############
# TODO: check the FIXME in awkward_listarray_getitem_next_array_advanced.
# TODO: setid should not be allowed on data that can be reached by multiple paths (which will break the ListArray ids above, unfortunately).
