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

def test_nonflat_slice():
    array = numpy.arange(2*3*5).reshape(2, 3, 5)
    numpyarray = awkward1.layout.NumpyArray(array)

    content = awkward1.layout.NumpyArray(array.reshape(-1))
    inneroffsets = awkward1.layout.Index64(numpy.array([0, 5, 10, 15, 20, 25, 30]))
    outeroffsets = awkward1.layout.Index64(numpy.array([0, 3, 6]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(outeroffsets, awkward1.layout.ListOffsetArray64(inneroffsets, content))
    listoffsetarray.setid()

    assert awkward1.tolist(array[[1, 0, 1, 1, 1, 0], [2, 0, 1, 1, 2, 0], [2, 4, 2, 4, 0, 1]]) == [27, 4, 22, 24, 25, 1]
    assert awkward1.tolist(array[[[1, 0], [1, 1], [1, 0]], [[2, 0], [1, 1], [2, 0]], [[2, 4], [2, 4], [0, 1]]]) == [[27, 4], [22, 24], [25, 1]]

    one = listoffsetarray[[1, 0, 1, 1, 1, 0], [2, 0, 1, 1, 2, 0], [2, 4, 2, 4, 0, 1]]
    assert awkward1.tolist(one) == [27, 4, 22, 24, 25, 1]
    assert numpy.asarray(one.id).tolist() == [
        [1, 2, 2],
        [0, 0, 4],
        [1, 1, 2],
        [1, 1, 4],
        [1, 2, 0],
        [0, 0, 1]]

    two = listoffsetarray[[[1, 0], [1, 1], [1, 0]], [[2, 0], [1, 1], [2, 0]], [[2, 4], [2, 4], [0, 1]]]
    assert awkward1.tolist(two) == [[27, 4], [22, 24], [25, 1]]
    assert numpy.asarray(two.content.id).tolist() == [
        [1, 2, 2],
        [0, 0, 4],
        [1, 1, 2],
        [1, 1, 4],
        [1, 2, 0],
        [0, 0, 1]]
    assert two.id is None

def test_newaxis():
    array = numpy.arange(2*3*5).reshape(2, 3, 5)
    numpyarray = awkward1.layout.NumpyArray(array)

    content = awkward1.layout.NumpyArray(array.reshape(-1))
    inneroffsets = awkward1.layout.Index64(numpy.array([0, 5, 10, 15, 20, 25, 30]))
    outeroffsets = awkward1.layout.Index64(numpy.array([0, 3, 6]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(outeroffsets, awkward1.layout.ListOffsetArray64(inneroffsets, content))

    assert awkward1.tolist(array[:, numpy.newaxis]) == [[[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]], [[[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]]

    assert awkward1.tolist(listoffsetarray[:, numpy.newaxis]) == [[[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]], [[[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]]

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

# Independent:
##############
# TODO: check the FIXME in awkward_listarray_getitem_next_array_advanced.
