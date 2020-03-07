# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_emptyarray():
    array = awkward1.layout.EmptyArray()
    assert awkward1.tolist(array.sizes(0)) == 0
    assert awkward1.tolist(array.sizes(1)) == []
    assert awkward1.tolist(array.sizes(2)) == []

def test_numpyarray():
    array = awkward1.layout.NumpyArray(numpy.arange(2*3*5*7).reshape(2, 3, 5, 7))
    assert array.sizes(0) == 2
    assert awkward1.tolist(array.sizes(1)) == [3, 3]
    assert awkward1.tolist(array.sizes(2)) == [[5, 5, 5], [5, 5, 5]]
    assert awkward1.tolist(array.sizes(3)) == [[[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]], [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]]]
    with pytest.raises(ValueError) as err:
        array.sizes(4)
    assert str(err.value) == "'axis' out of range for 'sizes'"

def test_regulararray():
    array = awkward1.layout.NumpyArray(numpy.arange(2*3*5*7).reshape(2, 3, 5, 7)).toRegularArray()
    assert array.sizes(0) == 2
    assert awkward1.tolist(array.sizes(1)) == [3, 3]
    assert awkward1.tolist(array.sizes(2)) == [[5, 5, 5], [5, 5, 5]]
    assert awkward1.tolist(array.sizes(3)) == [[[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]], [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]]]
    with pytest.raises(ValueError) as err:
        array.sizes(4)
    assert str(err.value) == "'axis' out of range for 'sizes'"

def test_listarray():
    content = awkward1.layout.NumpyArray(numpy.arange(2*3*5).reshape(5, 3, 2))
    starts = awkward1.layout.Index64(numpy.array([0, 3, 3], dtype=numpy.int64))
    stops = awkward1.layout.Index64(numpy.array([3, 3, 5], dtype=numpy.int64))
    array = awkward1.layout.ListArray64(starts, stops, content)
    assert awkward1.tolist(array) == [
        [[[ 0,  1], [ 2,  3], [ 4,  5]],
         [[ 6,  7], [ 8,  9], [10, 11]],
         [[12, 13], [14, 15], [16, 17]]],
        [],
        [[[18, 19], [20, 21], [22, 23]],
         [[24, 25], [26, 27], [28, 29]]]]

    assert awkward1.tolist(array.sizes(0)) == 3
    assert awkward1.tolist(array.sizes(1)) == [3, 0, 2]
    assert awkward1.tolist(array.sizes(2)) == [[3, 3, 3], [], [3, 3]]
    assert awkward1.tolist(array.sizes(3)) == [[[2, 2, 2], [2, 2, 2], [2, 2, 2]], [], [[2, 2, 2], [2, 2, 2]]]
    with pytest.raises(ValueError) as err:
        array.sizes(4)
    assert str(err.value) == "'axis' out of range for 'sizes'"

def test_listoffsetarray():
    content = awkward1.layout.NumpyArray(numpy.arange(2*3*5).reshape(5, 3, 2))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64))
    array = awkward1.layout.ListOffsetArray64(offsets, content)
    assert awkward1.tolist(array) == [
        [[[ 0,  1], [ 2,  3], [ 4,  5]],
         [[ 6,  7], [ 8,  9], [10, 11]],
         [[12, 13], [14, 15], [16, 17]]],
        [],
        [[[18, 19], [20, 21], [22, 23]],
         [[24, 25], [26, 27], [28, 29]]]]

    assert awkward1.tolist(array.sizes(0)) == 3
    assert awkward1.tolist(array.sizes(1)) == [3, 0, 2]
    assert awkward1.tolist(array.sizes(2)) == [[3, 3, 3], [], [3, 3]]
    assert awkward1.tolist(array.sizes(3)) == [[[2, 2, 2], [2, 2, 2], [2, 2, 2]], [], [[2, 2, 2], [2, 2, 2]]]
    with pytest.raises(ValueError) as err:
        array.sizes(4)
    assert str(err.value) == "'axis' out of range for 'sizes'"

def test_indexedarray():
    content = awkward1.layout.NumpyArray(numpy.arange(2*3*5).reshape(5, 3, 2))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64))
    listarray = awkward1.layout.ListOffsetArray64(offsets, content)
    index = awkward1.layout.Index64(numpy.array([2, 2, 1, 0], dtype=numpy.int64))
    array = awkward1.layout.IndexedArray64(index, listarray)
    assert awkward1.tolist(array) == [
        [[[18, 19], [20, 21], [22, 23]],
         [[24, 25], [26, 27], [28, 29]]],
        [[[18, 19], [20, 21], [22, 23]],
         [[24, 25], [26, 27], [28, 29]]],
        [],
        [[[ 0,  1], [ 2,  3], [ 4,  5]],
         [[ 6,  7], [ 8,  9], [10, 11]],
         [[12, 13], [14, 15], [16, 17]]]]

    assert awkward1.tolist(array.sizes(0)) == 4
    assert awkward1.tolist(array.sizes(1)) == [2, 2, 0, 3]
    assert awkward1.tolist(array.sizes(2)) == [[3, 3], [3, 3], [], [3, 3, 3]]
    assert awkward1.tolist(array.sizes(3)) == [[[2, 2, 2], [2, 2, 2]], [[2, 2, 2], [2, 2, 2]], [], [[2, 2, 2], [2, 2, 2], [2, 2, 2]]]

    with pytest.raises(ValueError) as err:
        array.sizes(4)
    assert str(err.value) == "'axis' out of range for 'sizes'"

def test_indexedoptionarray():
    content = awkward1.layout.NumpyArray(numpy.arange(2*3*5).reshape(5, 3, 2))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64))
    listarray = awkward1.layout.ListOffsetArray64(offsets, content)
    index = awkward1.layout.Index64(numpy.array([2, -1, 2, 1, -1, 0], dtype=numpy.int64))
    array = awkward1.layout.IndexedOptionArray64(index, listarray)
    assert awkward1.tolist(array) == [
        [[[18, 19], [20, 21], [22, 23]],
         [[24, 25], [26, 27], [28, 29]]],
        None,
        [[[18, 19], [20, 21], [22, 23]],
         [[24, 25], [26, 27], [28, 29]]],
        [],
        None,
        [[[ 0,  1], [ 2,  3], [ 4,  5]],
         [[ 6,  7], [ 8,  9], [10, 11]],
         [[12, 13], [14, 15], [16, 17]]]]

    print(array.sizes(1))
    raise Exception

    # assert awkward1.tolist(array.sizes(0)) == 4
    # assert awkward1.tolist(array.sizes(1)) == [2, 2, 0, 3]
    # assert awkward1.tolist(array.sizes(2)) == [[3, 3], [3, 3], [], [3, 3, 3]]
    # assert awkward1.tolist(array.sizes(3)) == [[[2, 2, 2], [2, 2, 2]], [[2, 2, 2], [2, 2, 2]], [], [[2, 2, 2], [2, 2, 2], [2, 2, 2]]]
    #
    # with pytest.raises(ValueError) as err:
    #     array.sizes(4)
    # assert str(err.value) == "'axis' out of range for 'sizes'"
