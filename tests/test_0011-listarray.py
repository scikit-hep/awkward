# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

py27 = (sys.version_info[0] < 3)

content  = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
starts1  = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6]))
stops1   = awkward1.layout.Index64(numpy.array([3, 3, 5, 6, 9]))
offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
starts2  = awkward1.layout.Index64(numpy.array([0, 2, 3, 3]))
stops2   = awkward1.layout.Index64(numpy.array([2, 3, 3, 5]))
offsets2 = awkward1.layout.Index64(numpy.array([0, 2, 3, 3, 5]))

def test_listarray_basic():
    array1 = awkward1.layout.ListArray64(starts1, stops1, content)
    array2 = awkward1.layout.ListArray64(starts2, stops2, array1)
    assert awkward1.to_list(array1) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert awkward1.to_list(array1[2]) == [4.4, 5.5]
    assert awkward1.to_list(array1[1:-1]) == [[], [4.4, 5.5], [6.6]]
    assert awkward1.to_list(array2) == [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9]]]
    assert awkward1.to_list(array2[1]) == [[4.4, 5.5]]
    assert awkward1.to_list(array2[1:-1]) == [[[4.4, 5.5]], []]

def test_listoffsetarray_basic():
    array1 = awkward1.layout.ListOffsetArray64(offsets1, content)
    array2 = awkward1.layout.ListOffsetArray64(offsets2, array1)
    assert awkward1.to_list(array1) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert awkward1.to_list(array1[2]) == [4.4, 5.5]
    assert awkward1.to_list(array1[1:-1]) == [[], [4.4, 5.5], [6.6]]
    assert awkward1.to_list(array2) == [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9]]]
    assert awkward1.to_list(array2[1]) == [[4.4, 5.5]]
    assert awkward1.to_list(array2[1:-1]) == [[[4.4, 5.5]], []]

def test_listarray_at():
    array1 = awkward1.layout.ListArray64(starts1, stops1, content)
    array2 = awkward1.layout.ListArray64(starts2, stops2, array1)
    assert awkward1.to_list(array1[2]) == [4.4, 5.5]
    assert awkward1.to_list(array1[2,]) == [4.4, 5.5]
    assert awkward1.to_list(array1[2, 1:]) == [5.5]
    assert awkward1.to_list(array1[2:, 0]) == [4.4, 6.6, 7.7]
    assert awkward1.to_list(array1[2:, -1]) == [5.5, 6.6, 9.9]

def test_listoffsetarray_at():
    array1 = awkward1.layout.ListOffsetArray64(offsets1, content)
    array2 = awkward1.layout.ListOffsetArray64(offsets2, array1)
    assert awkward1.to_list(array1[2,]) == [4.4, 5.5]
    assert awkward1.to_list(array1[2, 1:]) == [5.5]
    assert awkward1.to_list(array1[2:, 0]) == [4.4, 6.6, 7.7]
    assert awkward1.to_list(array1[2:, -1]) == [5.5, 6.6, 9.9]

def test_listarray_slice():
    array1 = awkward1.layout.ListArray64(starts1, stops1, content)
    array2 = awkward1.layout.ListArray64(starts2, stops2, array1)
    assert awkward1.to_list(array1[1:-1]) == [[], [4.4, 5.5], [6.6]]
    assert awkward1.to_list(array1[1:-1,]) == [[], [4.4, 5.5], [6.6]]
    assert awkward1.to_list(array2[1:-1]) == [[[4.4, 5.5]], []]
    assert awkward1.to_list(array2[1:-1,]) == [[[4.4, 5.5]], []]

def test_listoffsetarray_slice():
    array1 = awkward1.layout.ListOffsetArray64(offsets1, content)
    array2 = awkward1.layout.ListOffsetArray64(offsets2, array1)
    assert awkward1.to_list(array1[1:-1]) == [[], [4.4, 5.5], [6.6]]
    assert awkward1.to_list(array1[1:-1,]) == [[], [4.4, 5.5], [6.6]]
    assert awkward1.to_list(array2[1:-1]) == [[[4.4, 5.5]], []]
    assert awkward1.to_list(array2[1:-1,]) == [[[4.4, 5.5]], []]

def test_listarray_slice_slice():
    array1 = awkward1.layout.ListArray64(starts1, stops1, content)
    array2 = awkward1.layout.ListArray64(starts2, stops2, array1)
    assert awkward1.to_list(array1[2:]) == [[4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert awkward1.to_list(array1[2:, 1:]) == [[5.5], [], [8.8, 9.9]]
    assert awkward1.to_list(array1[2:,:-1]) == [[4.4], [], [7.7, 8.8]]

def test_listoffsetarray_slice_slice():
    array1 = awkward1.layout.ListOffsetArray64(offsets1, content)
    array2 = awkward1.layout.ListOffsetArray64(offsets2, array1)
    assert awkward1.to_list(array1[2:]) == [[4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert awkward1.to_list(array1[2:, 1:]) == [[5.5], [], [8.8, 9.9]]
    assert awkward1.to_list(array1[2:,:-1]) == [[4.4], [], [7.7, 8.8]]

def test_listarray_ellipsis():
    array1 = awkward1.layout.ListArray64(starts1, stops1, content)
    array2 = awkward1.layout.ListArray64(starts2, stops2, array1)
    if not py27:
        assert awkward1.to_list(array1[Ellipsis, 1:]) == [[2.2, 3.3], [], [5.5], [], [8.8, 9.9]]
        assert awkward1.to_list(array2[Ellipsis, 1:]) == [[[2.2, 3.3], []], [[5.5]], [], [[], [8.8, 9.9]]]

def test_listoffsetarray_ellipsis():
    array1 = awkward1.layout.ListOffsetArray64(offsets1, content)
    array2 = awkward1.layout.ListOffsetArray64(offsets2, array1)
    if not py27:
        assert awkward1.to_list(array1[Ellipsis, 1:]) == [[2.2, 3.3], [], [5.5], [], [8.8, 9.9]]
        assert awkward1.to_list(array2[Ellipsis, 1:]) == [[[2.2, 3.3], []], [[5.5]], [], [[], [8.8, 9.9]]]

def test_listarray_array_slice():
    array1 = awkward1.layout.ListArray64(starts1, stops1, content)
    array2 = awkward1.layout.ListArray64(starts2, stops2, array1)
    assert awkward1.to_list(array2[[0, 0, 1, 1, 1, 0]]) == [[[1.1, 2.2, 3.3], []], [[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [[4.4, 5.5]], [[4.4, 5.5]], [[1.1, 2.2, 3.3], []]]
    assert awkward1.to_list(array2[[0, 0, 1, 1, 1, 0], :]) == [[[1.1, 2.2, 3.3], []], [[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [[4.4, 5.5]], [[4.4, 5.5]], [[1.1, 2.2, 3.3], []]]
    assert awkward1.to_list(array2[[0, 0, 1, 1, 1, 0], :, 1:]) == [[[2.2, 3.3], []], [[2.2, 3.3], []], [[5.5]], [[5.5]], [[5.5]], [[2.2, 3.3], []]]

def test_listoffsetarray_array_slice():
    array1 = awkward1.layout.ListOffsetArray64(offsets1, content)
    array2 = awkward1.layout.ListOffsetArray64(offsets2, array1)
    assert awkward1.to_list(array2[[0, 0, 1, 1, 1, 0]]) == [[[1.1, 2.2, 3.3], []], [[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [[4.4, 5.5]], [[4.4, 5.5]], [[1.1, 2.2, 3.3], []]]
    assert awkward1.to_list(array2[[0, 0, 1, 1, 1, 0], :]) == [[[1.1, 2.2, 3.3], []], [[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [[4.4, 5.5]], [[4.4, 5.5]], [[1.1, 2.2, 3.3], []]]
    assert awkward1.to_list(array2[[0, 0, 1, 1, 1, 0], :, 1:]) == [[[2.2, 3.3], []], [[2.2, 3.3], []], [[5.5]], [[5.5]], [[5.5]], [[2.2, 3.3], []]]

def test_listarray_array():
    array1 = awkward1.layout.ListArray64(starts1, stops1, content)
    array2 = awkward1.layout.ListArray64(starts2, stops2, array1)
    assert awkward1.to_list(array1[numpy.array([2, 0, 0, 1, -1])]) == [[4.4, 5.5], [1.1, 2.2, 3.3], [1.1, 2.2, 3.3], [], [7.7, 8.8, 9.9]]
    assert awkward1.to_list(array1[numpy.array([2, 0, 0, -1]), numpy.array([1, 1, 0, 0])]) == [5.5, 2.2, 1.1, 7.7]

    content_deep = awkward1.layout.NumpyArray(numpy.array([[0, 0], [1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60], [7, 70], [8, 80]]))
    starts1_deep = awkward1.layout.Index64(numpy.array([0, 3, 6]))
    stops1_deep = awkward1.layout.Index64(numpy.array([3, 6, 9]))
    array1_deep = awkward1.layout.ListArray64(starts1_deep, stops1_deep, content_deep)

    assert awkward1.to_list(array1_deep) == [[[0, 0], [1, 10], [2, 20]], [[3, 30], [4, 40], [5, 50]], [[6, 60], [7, 70], [8, 80]]]
    s = (numpy.array([2, 0, 0, -1]), numpy.array([1, 1, 0, 0]), numpy.array([0, 1, 0, 1]))
    assert numpy.array([[[0, 0], [1, 10], [2, 20]], [[3, 30], [4, 40], [5, 50]], [[6, 60], [7, 70], [8, 80]]])[s].tolist() == awkward1.to_list(array1_deep[s])

    s = (numpy.array([2, 0, 0, -1]), numpy.array([1, 1, 0, 0]), slice(1, None))
    assert numpy.array([[[0, 0], [1, 10], [2, 20]], [[3, 30], [4, 40], [5, 50]], [[6, 60], [7, 70], [8, 80]]])[s].tolist() == awkward1.to_list(array1_deep[s])

def test_listoffsetarray_array():
    array1 = awkward1.layout.ListOffsetArray64(offsets1, content)
    array2 = awkward1.layout.ListOffsetArray64(offsets2, array1)
    assert awkward1.to_list(array1[numpy.array([2, 0, 0, 1, -1])]) == [[4.4, 5.5], [1.1, 2.2, 3.3], [1.1, 2.2, 3.3], [], [7.7, 8.8, 9.9]]
    assert awkward1.to_list(array1[numpy.array([2, 0, 0, -1]), numpy.array([1, 1, 0, 0])]) == [5.5, 2.2, 1.1, 7.7]

    content_deep = awkward1.layout.NumpyArray(numpy.array([[0, 0], [1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60], [7, 70], [8, 80]]))
    starts1_deep = awkward1.layout.Index64(numpy.array([0, 3, 6]))
    stops1_deep = awkward1.layout.Index64(numpy.array([3, 6, 9]))
    array1_deep = awkward1.layout.ListArray64(starts1_deep, stops1_deep, content_deep)

    assert awkward1.to_list(array1_deep) == [[[0, 0], [1, 10], [2, 20]], [[3, 30], [4, 40], [5, 50]], [[6, 60], [7, 70], [8, 80]]]
    s = (numpy.array([2, 0, 0, -1]), numpy.array([1, 1, 0, 0]), numpy.array([0, 1, 0, 1]))
    assert numpy.array([[[0, 0], [1, 10], [2, 20]], [[3, 30], [4, 40], [5, 50]], [[6, 60], [7, 70], [8, 80]]])[s].tolist() == awkward1.to_list(array1_deep[s])

    s = (numpy.array([2, 0, 0, -1]), numpy.array([1, 1, 0, 0]), slice(1, None))
    assert numpy.array([[[0, 0], [1, 10], [2, 20]], [[3, 30], [4, 40], [5, 50]], [[6, 60], [7, 70], [8, 80]]])[s].tolist() == awkward1.to_list(array1_deep[s])
