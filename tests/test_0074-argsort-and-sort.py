# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_sort_emptyarray():
    array = awkward1.layout.EmptyArray()
    #FIXME assert awkward1.to_list(array.sort(0, True, False)) == []
    #FIXME assert awkward1.to_list(array.argsort(0, True, False)) == []

def test_sort_numpyarray():
    array = awkward1.layout.NumpyArray(numpy.array([3.3, 2.2, 1.1, 5.5, 4.4]))
    assert awkward1.to_list(array.argsort(0, True, False)) == [2, 1, 0, 4, 3]
    assert awkward1.to_list(array.argsort(0, False, False)) == [3, 4, 0, 1, 2]

    assert awkward1.to_list(array.sort(0, True, False)) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.to_list(array.sort(0, False, False)) == [5.5, 4.4, 3.3, 2.2, 1.1]

    array = awkward1.layout.NumpyArray(numpy.array([[3.3, 2.2, 4.4], [1.1, 5.5, 3.3]]))
    assert awkward1.to_list(array.sort(1, True, False)) == [[2.2, 3.3, 4.4], [1.1, 3.3, 5.5]]
    assert awkward1.to_list(array.argsort(1, True, False)) == [[1, 0, 2], [0, 2, 1]]
    with pytest.raises(ValueError) as err:
        array.sort(2, True, False)
    assert str(err.value) == "axis=2 exceeds the depth of the nested list structure (which is 2)"

# def test_3d():
#     array = awkward1.layout.NumpyArray(numpy.array([
#         [[ 3.3,  2.2,  5.5,  1.1,  4.4],
#          [ 4.4,  2.2,  1.1,  3.3,  5.5],
#          [ 2.2,  1.1,  4.4,  3.3,  5.5]],
#         [[-3.3,  2.2, -5.5,  1.1,  4.4],
#          [ 4.4, -2.2,  1.1, -3.3,  5.5],
#          [ 2.2,  1.1,  4.4,  3.3, -5.5]]]))
#     # assert awkward1.to_list(array) == [
#     #     [1, 0, 1, 0, 0],
#     #     [0, 1, 0, 1, 0],
#     #     [0, 0, 0, 0, 1]]
#     assert awkward1.to_list(array.argsort(1, True, False)) == [
#         [2, 2, 1, 0, 0],
#         [0, 1, 0, 1, 2]]
#     # assert awkward1.to_list(array.argsort(2, True, False)) == [
#     #     [3, 2, 1],
#     #     [2, 3, 4]]

def test_sort_indexoffsetarray():
    array = awkward1.Array([[2.2, 1.1, 3.3], [], [4.4, 5.5], [5.5], [-4.4, -5.5, -6.6]]).layout
    assert awkward1.to_list(array.sort(1, True, False)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [5.5], [-6.6, -5.5, -4.4]]
    assert awkward1.to_list(array.argsort(1, True, False)) == [[1, 0, 2], [], [0, 1], [0], [2, 1, 0]]
    #FIXME assert awkward1.to_list(array.argsort(0, True, False)) == [9, 8, 7, 1, 0, 2, 4, 5, 6, 3]

    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    index1 = awkward1.layout.Index32(numpy.array([1, 2, 3, 4], dtype=numpy.int32))
    indexedarray1 = awkward1.layout.IndexedArray32(index1, content)
    assert awkward1.to_list(indexedarray1.sort(0, True, False)) == [2.2, 3.3, 4.4, 5.5]

    index2 = awkward1.layout.Index64(numpy.array([1, 2, 3], dtype=numpy.int64))
    indexedarray2 = awkward1.layout.IndexedArray64(index2, indexedarray1)
    assert awkward1.to_list(indexedarray2.sort(0, False, False)) == [5.5, 4.4, 3.3]

    index3 = awkward1.layout.Index32(numpy.array([1, 2], dtype=numpy.int32))
    indexedarray3 = awkward1.layout.IndexedArray32(index3, indexedarray2)
    assert awkward1.to_list(indexedarray3.sort(0, True, False)) == [4.4, 5.5]
