# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_EmptyArray():
    array = awkward1.layout.EmptyArray()
    assert awkward1.to_list(array.sort(0, True, False)) == []
    assert awkward1.to_list(array.argsort(0, True, False)) == []

def test_NumpyArray():
    array = awkward1.layout.NumpyArray(numpy.array([3.3, 2.2, 1.1, 5.5, 4.4]))
    assert awkward1.to_list(array.argsort(0, True, False)) == [2, 1, 0, 4, 3]
    assert awkward1.to_list(array.argsort(0, False, False)) == [3, 4, 0, 1, 2]

    assert awkward1.to_list(array.sort(0, True, False)) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.to_list(array.sort(0, False, False)) == [5.5, 4.4, 3.3, 2.2, 1.1]

    array2 = awkward1.layout.NumpyArray(numpy.array([[3.3, 2.2, 4.4],
                                                    [1.1, 5.5, 3.3]]))
    # np.sort(array2, axis=1)
    # array([[2.2, 3.3, 4.4],
    #        [1.1, 3.3, 5.5]])
    assert awkward1.to_list(array2.sort(1, True, False)) == [
        [2.2, 3.3, 4.4],
        [1.1, 3.3, 5.5]]

    # np.sort(array2, axis=0)
    # array([[1.1, 2.2, 3.3],
    #        [3.3, 5.5, 4.4]])
    assert awkward1.to_list(array2.sort(0, True, False)) == [
        [1.1, 2.2, 3.3],
        [3.3, 5.5, 4.4]]

    assert awkward1.to_list(array2.argsort(1, True, False)) == [
        [1, 0, 2],
        [0, 2, 1]]
    assert awkward1.to_list(array2.argsort(0, True, False)) == [
        [1, 0, 1],
        [0, 1, 0]]

    with pytest.raises(ValueError) as err:
        array2.sort(2, True, False)
    assert str(err.value).startswith("axis=2 exceeds the depth of the nested list structure (which is 2)")

def test_IndexedOffsetArray():
    array = awkward1.Array([[  2.2, 1.1,   3.3 ],
                            [ None, None, None ],
                            [  4.4, None,  5.5 ],
                            [  5.5, None, None ],
                            [ -4.4, -5.5, -6.6 ]]).layout

    assert awkward1.to_list(array.sort(0, True, False)) == [
        [-4.4, -5.5, -6.6],
        [ 2.2,  1.1,  3.3],
        [ 4.4, None,  5.5],
        [ 5.5, None, None],
        [None, None, None]]

    assert awkward1.to_list(array.argsort(0, True, False)) == [
        [   3,    1,    2],
        [   0,    0,    0],
        [   1, None,    1],
        [   2, None, None],
        [None, None, None]]

    assert awkward1.to_list(array.sort(1, True, False)) == [
        [  1.1,  2.2,   3.3],
        [ None, None, None ],
        [  4.4,  5.5, None ],
        [  5.5, None, None ],
        [ -6.6, -5.5, -4.4 ]]

    assert awkward1.to_list(array.argsort(1, True, False)) == [
        [    1,    0,    2 ],
        [ None, None, None ],
        [    0,    1, None ],
        [    0, None, None ],
        [    2,    1,    0 ]]

    assert awkward1.to_list(array.sort(1, False, False)) == [
        [  3.3,  2.2,  1.1 ],
        [ None, None, None ],
        [  5.5,  4.4, None ],
        [  5.5, None, None ],
        [ -4.4, -5.5, -6.6 ]]

    assert awkward1.to_list(array.argsort(1, False, False)) == [
        [    2,    0,    1 ],
        [ None, None, None ],
        [    1,    0, None ],
        [    0, None, None ],
        [    0,    1,    2 ]]

    array3 = awkward1.Array([[ 2.2,  1.1,  3.3],
                             [],
                             [ 4.4,  5.5 ],
                             [ 5.5 ],
                             [-4.4, -5.5, -6.6]]).layout

    assert awkward1.to_list(array3.sort(1, False, False)) == [
        [  3.3,  2.2,  1.1 ],
        [],
        [  5.5,  4.4 ],
        [  5.5 ],
        [ -4.4, -5.5, -6.6 ]]

    assert awkward1.to_list(array3.sort(0, True, False)) == [
        [ -4.4, -5.5, -6.6 ],
        [],
        [  2.2,  1.1 ],
        [  4.4 ],
        [  5.5, 5.5, 3.3 ]]

#FIXME: Based on Numpy list sorting:
#
# array([list([2.2, 1.1, 3.3]), list([]), list([4.4, 5.5]), list([5.5]),
#        list([-4.4, -5.5, -6.6])], dtype=object)
# np.sort(array, axis=0)
# array([list([]), list([-4.4, -5.5, -6.6]), list([2.2, 1.1, 3.3]),
#        list([4.4, 5.5]), list([5.5])], dtype=object)
#
# the result should be:
    #
    # [[ -4.4, -5.5, -6.6 ],
    #  [  2.2,  1.1,  3.3 ],
    #  [  4.4,  5.5 ],
    #  [  5.5 ],
    #  []]

# This can be done following the steps: pad, sort,
# and dropna to strip off the None's
#
    array4 = array3.rpad(3, 1)
    assert awkward1.to_list(array4) == [
        [2.2, 1.1, 3.3],
        [None, None, None],
        [4.4, 5.5, None],
        [5.5, None, None],
        [-4.4, -5.5, -6.6]]

    array5 = array4.sort(0, True, False)
    assert awkward1.to_list(array5) == [
        [-4.4, -5.5, -6.6],
        [2.2, 1.1, 3.3],
        [4.4, 5.5, None],
        [5.5, None, None],
        [None, None, None]]

    array4 = array3.rpad(5, 1)
    assert awkward1.to_list(array4) == [
        [ 2.2,  1.1,  3.3, None, None],
        [None, None, None, None, None],
        [ 4.4,  5.5, None, None, None],
        [ 5.5, None, None, None, None],
        [-4.4, -5.5, -6.6, None, None]]

    array5 = array4.sort(0, True, False)
    assert awkward1.to_list(array5) == [
        [-4.4, -5.5, -6.6, None, None],
        [ 2.2,  1.1,  3.3, None, None],
        [ 4.4,  5.5, None, None, None],
        [ 5.5, None, None, None, None],
        [None, None, None, None, None]]

    array5 = array4.argsort(0, True, False)
    assert awkward1.to_list(array5) == [
        [   3,    2,    1, None, None],
        [   0,    0,    0, None, None],
        [   1,    1, None, None, None],
        [   2, None, None, None, None],
        [None, None, None, None, None]]

# FIXME: implement dropna to strip off the None's
#
    # array6 = array5.dropna(0)
    # assert awkward1.to_list(array6) == [
    #     [ -4.4, -5.5, -6.6 ],
    #     [  2.2,  1.1,  3.3 ],
    #     [  4.4,  5.5 ],
    #     [  5.5 ],
    #     []]

    # FIXME: assert awkward1.to_list(array.argsort(1, True, False)) == [[1, 0, 2], [], [0, 1], [0], [2, 1, 0]]
    # [[1, 0, 2], [None, None, None], [-3, -2, None], [-4, None, None], [-4, -5, -6]]
    # FIXME: assert awkward1.to_list(array.argsort(0, True, False)) == [[3, 0, 1, 2], [2, 0, 1], [1, 0]]
    # [[3, 0, -2], [0, -1, -4], [1, None, -3], [2, None, None], [None, None, None]]

    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    index1 = awkward1.layout.Index32(numpy.array([1, 2, 3, 4], dtype=numpy.int32))
    indexedarray1 = awkward1.layout.IndexedArray32(index1, content)
    assert awkward1.to_list(indexedarray1.argsort(0, True, False)) == [0, 1, 2, 3]

    index2 = awkward1.layout.Index64(numpy.array([1, 2, 3], dtype=numpy.int64))
    indexedarray2 = awkward1.layout.IndexedArray64(index2, indexedarray1)
    assert awkward1.to_list(indexedarray2.sort(0, False, False)) == [5.5, 4.4, 3.3]

    index3 = awkward1.layout.Index32(numpy.array([1, 2], dtype=numpy.int32))
    indexedarray3 = awkward1.layout.IndexedArray32(index3, indexedarray2)
    assert awkward1.to_list(indexedarray3.sort(0, True, False)) == [4.4, 5.5]

def test_3d():
    array = awkward1.layout.NumpyArray(numpy.array([
# axis 2:    0       1       2       3       4         # axis 1:
        [[  1.1,    2.2,    3.3,    4.4,    5.5 ],     # 0
         [  6.6,    7.7,    8.8,    9.9,   10.10],     # 1
         [ 11.11,  12.12,  13.13,  14.14,  15.15]],    # 2
        [[ -1.1,   -2.2,   -3.3,   -4.4,   -5.5],      # 3
         [ -6.6,   -7.7,   -8.8,   -9.9,  -10.1],      # 4
         [-11.11, -12.12, -13.13, -14.14, -15.15]]]))  # 5
    assert awkward1.to_list(array.argsort(2, True, False)) == [
        [[0, 1, 2, 3, 4],
         [0, 1, 2, 3, 4],
         [0, 1, 2, 3, 4]],

        [[4, 3, 2, 1, 0],
         [4, 3, 2, 1, 0],
         [4, 3, 2, 1, 0]]]

# np.sort(array, axis=2)
# array([[[  1.1 ,   2.2 ,   3.3 ,   4.4 ,   5.5 ],
#         [  6.6 ,   7.7 ,   8.8 ,   9.9 ,  10.1 ],
#         [ 11.11,  12.12,  13.13,  14.14,  15.15]],
#
#        [[ -5.5 ,  -4.4 ,  -3.3 ,  -2.2 ,  -1.1 ],
#         [-10.1 ,  -9.9 ,  -8.8 ,  -7.7 ,  -6.6 ],
#         [-15.15, -14.14, -13.13, -12.12, -11.11]]])

    assert awkward1.to_list(array.sort(2, True, False)) == [
        [[  1.1,    2.2,    3.3,    4.4,    5.5 ],
         [  6.6,    7.7,    8.8,    9.9,   10.10],
         [ 11.11,  12.12,  13.13,  14.14,  15.15]],

        [[ -5.5,   -4.4,   -3.3,   -2.2,   -1.1],
         [-10.1,   -9.9,   -8.8,   -7.7,   -6.6],
         [-15.15, -14.14, -13.13, -12.12, -11.11]]]

# np.argsort(array, 2)
# array([[[0, 1, 2, 3, 4],
#         [0, 1, 2, 3, 4],
#         [0, 1, 2, 3, 4]],
#
#        [[4, 3, 2, 1, 0],
#         [4, 3, 2, 1, 0],
#         [4, 3, 2, 1, 0]]])
    assert awkward1.to_list(array.argsort(2, True, False)) == [
       [[0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4]],

       [[4, 3, 2, 1, 0],
        [4, 3, 2, 1, 0],
        [4, 3, 2, 1, 0]]]

# np.sort(array, axis=1)
# array([[[  1.1 ,   2.2 ,   3.3 ,   4.4 ,   5.5 ],
#         [  6.6 ,   7.7 ,   8.8 ,   9.9 ,  10.1 ],
#         [ 11.11,  12.12,  13.13,  14.14,  15.15]],
#
#        [[-11.11, -12.12, -13.13, -14.14, -15.15],
#         [ -6.6 ,  -7.7 ,  -8.8 ,  -9.9 , -10.1 ],
#         [ -1.1 ,  -2.2 ,  -3.3 ,  -4.4 ,  -5.5 ]]])

    assert awkward1.to_list(array.sort(1, True, False)) == [
    [[  1.1,    2.2,    3.3,    4.4,    5.5],
     [  6.6,    7.7,    8.8,    9.9,   10.1],
     [ 11.11,  12.12,  13.13,  14.14,  15.15]],

    [[-11.11, -12.12, -13.13, -14.14, -15.15],
     [ -6.6,   -7.7,   -8.8,   -9.9,  -10.1],
     [ -1.1,   -2.2,   -3.3,   -4.4,   -5.5]]]

    assert awkward1.to_list(array.sort(1, False, False)) == [
    [[ 11.11,  12.12,  13.13,  14.14,  15.15],
     [  6.6,    7.7,    8.8,    9.9,   10.1],
     [  1.1,    2.2,    3.3,    4.4,    5.5]],

    [[ -1.1,   -2.2,   -3.3,   -4.4,   -5.5],
     [ -6.6,   -7.7,   -8.8,   -9.9,  -10.1],
     [-11.11, -12.12, -13.13, -14.14, -15.15]]]

# np.argsort(array, 1)
# array([[[0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1],
#         [2, 2, 2, 2, 2]],
#
#        [[2, 2, 2, 2, 2],
#         [1, 1, 1, 1, 1],
#         [0, 0, 0, 0, 0]]])

    assert awkward1.to_list(array.argsort(1, True, False)) == [
        [[0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2]],

        [[2, 2, 2, 2, 2],
         [1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0]]]

# np.sort(array, axis=0)
# array([[[ -1.1 ,  -2.2 ,  -3.3 ,  -4.4 ,  -5.5 ],
#         [ -6.6 ,  -7.7 ,  -8.8 ,  -9.9 , -10.1 ],
#         [-11.11, -12.12, -13.13, -14.14, -15.15]],
#
#        [[  1.1 ,   2.2 ,   3.3 ,   4.4 ,   5.5 ],
#         [  6.6 ,   7.7 ,   8.8 ,   9.9 ,  10.1 ],
#         [ 11.11,  12.12,  13.13,  14.14,  15.15]]])

    assert awkward1.to_list(array.sort(0, True, False)) == [
    [[ -1.1,   -2.2,   -3.3,   -4.4,   -5.5],
     [ -6.6,   -7.7,   -8.8,   -9.9,  -10.1],
     [-11.11, -12.12, -13.13, -14.14, -15.15]],

    [[  1.1,    2.2,    3.3,    4.4,    5.5],
     [  6.6,    7.7,    8.8,    9.9,   10.1],
     [ 11.11,  12.12,  13.13,  14.14,  15.15]]]

# np.argsort(array, 0)
# array([[[1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1]],
#
#        [[0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0]]])

    assert awkward1.to_list(array.argsort(0, True, False)) == [
       [[1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]],

       [[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]]]

def test_RecordArray():
    array = awkward1.Array([{"x": 0.0, "y": []},
                            {"x": 1.1, "y": [1]},
                            {"x": 2.2, "y": [2, 2]},
                            {"x": 3.3, "y": [3, 3, 3]},
                            {"x": 4.4, "y": [4, 4, 4, 4]},
                            {"x": 5.5, "y": [5, 5, 5]},
                            {"x": 6.6, "y": [6, 6]},
                            {"x": 7.7, "y": [7]},
                            {"x": 8.8, "y": []}])
    assert awkward1.to_list(array) == [
     {'x': 0.0, 'y': []},
     {'x': 1.1, 'y': [1]},
     {'x': 2.2, 'y': [2, 2]},
     {'x': 3.3, 'y': [3, 3, 3]},
     {'x': 4.4, 'y': [4, 4, 4, 4]},
     {'x': 5.5, 'y': [5, 5, 5]},
     {'x': 6.6, 'y': [6, 6]},
     {'x': 7.7, 'y': [7]},
     {'x': 8.8, 'y': []}]

    assert awkward1.to_list(array.layout.sort(-1, True, False)) == {
      'x': [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8],
      'y': [[], [1], [2, 2], [3, 3, 3], [4, 4, 4, 4], [5, 5, 5], [6, 6], [7], []]}

    assert awkward1.to_list(array.layout.sort(-1, False, False)) == {
      'x': [8.8, 7.7, 6.6, 5.5, 4.4, 3.3, 2.2, 1.1, 0.0],
      'y': [[], [1], [2, 2], [3, 3, 3], [4, 4, 4, 4], [5, 5, 5], [6, 6], [7], []]}

    assert awkward1.to_list(array.layout.argsort(-1, True, False)) == {
      'x': [0, 1, 2, 3, 4, 5, 6, 7, 8],
      'y': [[], [0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2], [0, 1], [0], []]}

    assert awkward1.to_list(array.x.layout.argsort(0, True, False)) == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert awkward1.to_list(array.x.layout.argsort(0, False, False)) == [8, 7, 6, 5, 4, 3, 2, 1, 0]

    array_y = array.y
    assert awkward1.to_list(array_y) == [[], [1], [2, 2], [3, 3, 3], [4, 4, 4, 4], [5, 5, 5], [6, 6], [7], []]
    assert awkward1.to_list(array.y.layout.argsort(0, True, False)) == [
        [],
        [0],
        [1, 0],
        [2, 1, 0],
        [3, 2, 1, 0],
        [4, 3, 2],
        [5, 4],
        [6],
        []]

    assert awkward1.to_list(array.y.layout.argsort(1, True, True)) == [
        [],
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 2],
        [0, 1],
        [0],
        []]

def test_ByteMaskedArray():
    content = awkward1.from_iter([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False)
    mask = awkward1.layout.Index8(numpy.array([0, 0, 1, 1, 0], dtype=numpy.int8))
    array = awkward1.layout.ByteMaskedArray(mask, content, valid_when=False)
    assert awkward1.to_list(array.argsort(0, True, False)) == [
        [0, 0, 0],
        [],
        [1, 1, 1, 0]]

    assert awkward1.to_list(array.sort(0, True, False)) == [
        [0.0, 1.1, 2.2],
        [],
        [6.6, 7.7, 8.8, 9.9]]

    assert awkward1.to_list(array.sort(0, False, False)) == [
        [6.6, 7.7, 8.8],
        [],
        [0.0, 1.1, 2.2, 9.9]]

    assert awkward1.to_list(array.argsort(1, True, False)) == [
        [0, 1, 2],
        [],
        None,
        None,
        [0, 1, 2, 3]]

    assert awkward1.to_list(array.sort(1, False, False)) ==  [
        [2.2, 1.1, 0.0],
        [],
        None,
        None,
        [9.9, 8.8, 7.7, 6.6]]

def test_UnionArray():
    content0 = awkward1.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    content1 = awkward1.from_iter(["one", "two", "three", "four", "five"], highlevel=False)
    tags = awkward1.layout.Index8(numpy.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=numpy.int8))
    index = awkward1.layout.Index32(numpy.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=numpy.int32))
    array = awkward1.layout.UnionArray8_32(tags, index, [content0, content1])

    with pytest.raises(ValueError) as err:
        array.sort(1, True, False)
    assert str(err.value).startswith("cannot sort UnionArray8_32")

def test_sort_strings():
    content1 = awkward1.from_iter(["one", "two", "three", "four", "five"], highlevel=False)
    assert awkward1.to_list(content1) == ['one', 'two', 'three', 'four', 'five']

    assert awkward1.to_list(content1.sort(0, True, False)) == ['five', 'four', 'one', 'three', 'two']
    assert awkward1.to_list(content1.sort(0, False, False)) == ['two', 'three', 'one', 'four', 'five']

    assert awkward1.to_list(content1.sort(1, True, False)) == ['five', 'four', 'one', 'three', 'two']
    assert awkward1.to_list(content1.sort(1, False, False)) == ['two', 'three', 'one', 'four', 'five']

def test_sort_bytestrings():
    array = awkward1.from_iter([b"one", b"two", b"three", b"two", b"two", b"one", b"three"], highlevel=False)
    assert awkward1.to_list(array) == [b'one', b'two', b'three', b'two', b'two', b'one', b'three']

    assert awkward1.to_list(array.sort(0, True, False)) == [b'one', b'one', b'three', b'three', b'two', b'two', b'two']
