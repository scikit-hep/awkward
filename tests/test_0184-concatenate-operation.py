# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_list_offset_array_megre():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    content2 = awkward1.layout.NumpyArray(numpy.array([999.999, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10], dtype=numpy.int64))
    array   = awkward1.layout.ListOffsetArray64(offsets, content)

    assert awkward1.to_list(array) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]

    # array1 = [[[1.1 2.2] [3.3]] [] [[4.4 5.5]] [[6.6 7.7 8.8] [] [9.9]]]
    # array2 = [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]
    offsets2 = awkward1.layout.Index64(numpy.array([1, 3, 4, 4, 6, 9, 9, 10], dtype=numpy.int64))
    array2   = awkward1.layout.ListOffsetArray64(offsets2, content)
    array22   = awkward1.layout.ListOffsetArray64(offsets2, content2)

    assert awkward1.to_list(array2) == [[1.1, 2.2], [3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8], [], [9.9]]
    assert awkward1.to_list(array.merge(array2, 0)) == [[0.0, 1.1, 2.2],
                                                        [],
                                                        [3.3, 4.4],
                                                        [5.5],
                                                        [6.6, 7.7, 8.8, 9.9],
                                                        [1.1, 2.2],
                                                        [3.3],
                                                        [],
                                                        [4.4, 5.5],
                                                        [6.6, 7.7, 8.8],
                                                        [],
                                                        [9.9]]

    assert awkward1.to_list(array.merge(array2, 1)) == [[0.0, 1.1, 2.2, 1.1, 2.2],
                                                        [3.3],
                                                        [3.3, 4.4],
                                                        [5.5, 4.4, 5.5],
                                                        [6.6, 7.7, 8.8, 9.9, 6.6, 7.7, 8.8],
                                                        [],
                                                        [9.9]]

    with pytest.raises(ValueError) as err:
        assert awkward1.to_list(array.merge(array2, 2))
    assert str(err.value).startswith("axis out of range for concatenate")

    offsets3 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64))
    array3   = awkward1.layout.ListOffsetArray64(offsets3, array)

    assert awkward1.to_list(array3) == [[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5], [6.6, 7.7, 8.8, 9.9]]]

    offsets4 = awkward1.layout.Index64(numpy.array([1, 3, 4, 4, 6, 7, 7], dtype=numpy.int64))
    array4   = awkward1.layout.ListOffsetArray64(offsets4, array22)

    assert awkward1.to_list(array4) == [[[0.33], []], [[0.44, 0.55]], [], [[0.66, 0.77, 0.88], []], [[0.99]], []]

    assert awkward1.to_list(array3.merge(array4, 0)) == [[[0.0, 1.1, 2.2], [], [3.3, 4.4]],
                                                         [],
                                                         [[5.5], [6.6, 7.7, 8.8, 9.9]],
                                                         [[0.33], []],
                                                         [[0.44, 0.55]],
                                                         [],
                                                         [[0.66, 0.77, 0.88], []],
                                                         [[0.99]],
                                                         []]

    assert awkward1.to_list(array3.merge(array4, 1)) == [[[0.0, 1.1, 2.2], [], [3.3, 4.4], [0.33], []],
                                                         [[0.44, 0.55]],
                                                         [[5.5], [6.6, 7.7, 8.8, 9.9]],
                                                         [[0.66, 0.77, 0.88], []],
                                                         [[0.99]],
                                                         []]

    # [[0.0, 1.1, 2.2, 3.3, 4.4], [], [5.5, 6.6, 7.7, 8.8, 9.9]]
    # [[3.3], [4.4, 5.5], [], [6.6, 7.7, 8.8], [9.9], []]
    # [[0.0, 1.1, 2.2, 3.3, 4.4, 3.3],\n [4.4, 5.5],\n [5.5, 6.6, 7.7, 8.8, 9.9],\n [6.6, 7.7, 8.8],\n [9.9],\n []]
    assert awkward1.to_list(array3.merge(array4, 2)) == [[[0.0, 1.1, 2.2], [], [3.3, 4.4], [0.33], []],
                                                         [[0.44, 0.55]],
                                                         [[5.5], [6.6, 7.7, 8.8, 9.9]],
                                                         [[0.66, 0.77, 0.88], []],
                                                         [[0.99]],
                                                         []]

def test_list_array_merge():
    one = awkward1.Array([[1, 2, 3], [], [4, 5]]).layout
    two = awkward1.Array([[1.1, 2.2], [3.3, 4.4]]).layout

    one = awkward1.layout.ListArray64(one.starts, one.stops, one.content)
    two = awkward1.layout.ListArray64(two.starts, two.stops, two.content)
    assert awkward1.to_list(one.merge(two, 0)) == [[1, 2, 3], [], [4, 5], [1.1, 2.2], [3.3, 4.4]]
    assert awkward1.to_list(one.merge(two, 1)) == [[1, 2, 3, 1.1, 2.2], [3.3, 4.4], [4, 5]]

def test_records_merge():
    one = awkward1.Array([{"x": 1, "y": [1]}, {"x": 2, "y": [1, 2]}, {"x": 3, "y": [1, 2, 3]}]).layout
    two = awkward1.Array([{"y": [], "x": 4}, {"y": [3, 2, 1], "x": 5}]).layout
    assert awkward1.to_list(one.merge(two, 0)) == [{"x": 1, "y": [1]}, {"x": 2, "y": [1, 2]}, {"x": 3, "y": [1, 2, 3]},
                                                   {"y": [], "x": 4}, {"y": [3, 2, 1], "x": 5}]
    with pytest.raises(ValueError) as err:
        awkward1.to_list(one.merge(two, 1))
    assert str(err.value).startswith("arrays of records cannot be concatenated (but their contents can be; try a different 'axis')")

    with pytest.raises(ValueError) as err:
        awkward1.to_list(one.merge(two, 2))
    assert str(err.value).startswith("axis out of range for concatenate")

def test_indexed_array_merge():
    one = awkward1.Array([[1, 2, 3], [None, 4], None, [None, 5]]).layout
    two = awkward1.Array([6, 7, 8]).layout
    three = awkward1.Array([[6.6], [7.7, 8.8]]).layout

    assert awkward1.to_list(one.merge(two, 0)) == [[1, 2, 3], [None, 4], None, [None, 5], 6, 7, 8]
    assert awkward1.to_list(one.merge(three, 1)) == [[1, 2, 3, 6.6], [None, 4, 7.7, 8.8], [], [None, 5]]

def test_bytemasked_merge():
    one = awkward1.Array([1, 2, 3, 4, 5, 6]).mask[[True, True, False, True, False, True]].layout
    two = awkward1.Array([7, 99, 999, 8, 9]).mask[[True, False, False, True, True]].layout

    assert awkward1.to_list(one.merge(two, 0)) == [1, 2, None, 4, None, 6, 7, None, None, 8, 9]

    with pytest.raises(ValueError) as err:
        awkward1.to_list(one.merge(two, 1))
    assert str(err.value).startswith("axis out of range for concatenate")

def test_listoffsetarray_merge():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 9]))
    listoffsetarray1 = awkward1.layout.ListOffsetArray64(offsets1, content1)

    assert awkward1.to_list(listoffsetarray1) == [[1, 2, 3], [], [4, 5], [6, 7, 8, 9]]

    content2 = awkward1.layout.NumpyArray(numpy.array([100, 200, 300, 400, 500]))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 2, 4, 4, 5]))
    listoffsetarray2 = awkward1.layout.ListOffsetArray64(offsets2, content2)

    assert awkward1.to_list(listoffsetarray2) == [[100, 200], [300, 400], [], [500]]
    assert awkward1.to_list(listoffsetarray1.merge(listoffsetarray2, 0)) == [[1, 2, 3], [], [4, 5], [6, 7, 8, 9], [100, 200], [300, 400], [], [500]]
    # another_array = listoffsetarray1.merge(listoffsetarray2, 1)
    # assert awkward1.to_list(another_array) == [[1, 2, 3, 100, 200], [300, 400], [4, 5], [6, 7, 8, 9, 500]]

    # tags = awkward1.layout.Index8(numpy.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=numpy.int8))
    # index = awkward1.layout.Index32(numpy.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=numpy.int32))
    # array = awkward1.layout.UnionArray8_32(tags, index, [listoffsetarray1, listoffsetarray2])
    #assert awkward1.to_list(array.simplify()) == [[1, 2, 3, 100, 200], [300, 400], [4, 5], [6, 7, 8, 9, 500]]
    final_array = awkward1.layout.EmptyArray()

    for i in range(4) :
        tags = awkward1.layout.Index8(numpy.array([0], dtype=numpy.int8))
        index = awkward1.layout.Index32(numpy.array([i], dtype=numpy.int32))
        array = awkward1.layout.UnionArray8_32(tags, index, [listoffsetarray1, listoffsetarray2])
        tags1 = awkward1.layout.Index8(numpy.array([1], dtype=numpy.int8))
        index1 = awkward1.layout.Index32(numpy.array([i], dtype=numpy.int32))
        array1 = awkward1.layout.UnionArray8_32(tags1, index1, [listoffsetarray1, listoffsetarray2])

        another_array = array.merge(array1).flatten()
        final_array = final_array.merge(another_array)

    final_offsets = awkward1.layout.Index64(numpy.array([0 + 0, 3 + 2, 3 + 4, 5 + 4, 9 + 5])) # offsets1 + offsets2

    assert awkward1.to_list(awkward1.layout.ListOffsetArray64(final_offsets, final_array)) == [[1, 2, 3, 100, 200], [300, 400], [4, 5], [6, 7, 8, 9, 500]]

def test_numpyarray_merge():
    emptyarray = awkward1.layout.EmptyArray()

    np1 = numpy.arange(2*7*5).reshape(2, 7, 5)
    np2 = numpy.arange(3*7*5).reshape(3, 7, 5)
    ak1 = awkward1.layout.NumpyArray(np1)
    ak2 = awkward1.layout.NumpyArray(np2)

    assert awkward1.to_list(ak1) == [[[ 0,  1,  2,  3,  4],
                                      [ 5,  6,  7,  8,  9],
                                      [10, 11, 12, 13, 14],
                                      [15, 16, 17, 18, 19],
                                      [20, 21, 22, 23, 24],
                                      [25, 26, 27, 28, 29],
                                      [30, 31, 32, 33, 34]],

                                     [[35, 36, 37, 38, 39],
                                      [40, 41, 42, 43, 44],
                                      [45, 46, 47, 48, 49],
                                      [50, 51, 52, 53, 54],
                                      [55, 56, 57, 58, 59],
                                      [60, 61, 62, 63, 64],
                                      [65, 66, 67, 68, 69]]]

    assert awkward1.to_list(ak2) == [[[ 0,  1,  2,  3,  4],
                                      [ 5,  6,  7,  8,  9],
                                      [10, 11, 12, 13, 14],
                                      [15, 16, 17, 18, 19],
                                      [20, 21, 22, 23, 24],
                                      [25, 26, 27, 28, 29],
                                      [30, 31, 32, 33, 34]],

                                     [[35, 36, 37, 38, 39],
                                      [40, 41, 42, 43, 44],
                                      [45, 46, 47, 48, 49],
                                      [50, 51, 52, 53, 54],
                                      [55, 56, 57, 58, 59],
                                      [60, 61, 62, 63, 64],
                                      [65, 66, 67, 68, 69]],

                                     [[70, 71, 72, 73, 74],
                                      [75, 76, 77, 78, 79],
                                      [80, 81, 82, 83, 84],
                                      [85, 86, 87, 88, 89],
                                      [90, 91, 92, 93, 94],
                                      [95, 96, 97, 98, 99],
                                      [100, 101, 102, 103, 104]]]

    assert awkward1.to_list(awkward1.concatenate([np1, np2, np1, np2], 0)) == [[[0, 1, 2, 3, 4],
                                                                                [5, 6, 7, 8, 9],
                                                                                [10, 11, 12, 13, 14],
                                                                                [15, 16, 17, 18, 19],
                                                                                [20, 21, 22, 23, 24],
                                                                                [25, 26, 27, 28, 29],
                                                                                [30, 31, 32, 33, 34]],
                                                                               [[35, 36, 37, 38, 39],
                                                                                [40, 41, 42, 43, 44],
                                                                                [45, 46, 47, 48, 49],
                                                                                [50, 51, 52, 53, 54],
                                                                                [55, 56, 57, 58, 59],
                                                                                [60, 61, 62, 63, 64],
                                                                                [65, 66, 67, 68, 69]],
                                                                               [[0, 1, 2, 3, 4],
                                                                                [5, 6, 7, 8, 9],
                                                                                [10, 11, 12, 13, 14],
                                                                                [15, 16, 17, 18, 19],
                                                                                [20, 21, 22, 23, 24],
                                                                                [25, 26, 27, 28, 29],
                                                                                [30, 31, 32, 33, 34]],
                                                                               [[35, 36, 37, 38, 39],
                                                                                [40, 41, 42, 43, 44],
                                                                                [45, 46, 47, 48, 49],
                                                                                [50, 51, 52, 53, 54],
                                                                                [55, 56, 57, 58, 59],
                                                                                [60, 61, 62, 63, 64],
                                                                                [65, 66, 67, 68, 69]],
                                                                               [[70, 71, 72, 73, 74],
                                                                                [75, 76, 77, 78, 79],
                                                                                [80, 81, 82, 83, 84],
                                                                                [85, 86, 87, 88, 89],
                                                                                [90, 91, 92, 93, 94],
                                                                                [95, 96, 97, 98, 99],
                                                                                [100, 101, 102, 103, 104]],
                                                                               [[0, 1, 2, 3, 4],
                                                                                [5, 6, 7, 8, 9],
                                                                                [10, 11, 12, 13, 14],
                                                                                [15, 16, 17, 18, 19],
                                                                                [20, 21, 22, 23, 24],
                                                                                [25, 26, 27, 28, 29],
                                                                                [30, 31, 32, 33, 34]],
                                                                               [[35, 36, 37, 38, 39],
                                                                                [40, 41, 42, 43, 44],
                                                                                [45, 46, 47, 48, 49],
                                                                                [50, 51, 52, 53, 54],
                                                                                [55, 56, 57, 58, 59],
                                                                                [60, 61, 62, 63, 64],
                                                                                [65, 66, 67, 68, 69]],
                                                                               [[0, 1, 2, 3, 4],
                                                                                [5, 6, 7, 8, 9],
                                                                                [10, 11, 12, 13, 14],
                                                                                [15, 16, 17, 18, 19],
                                                                                [20, 21, 22, 23, 24],
                                                                                [25, 26, 27, 28, 29],
                                                                                [30, 31, 32, 33, 34]],
                                                                               [[35, 36, 37, 38, 39],
                                                                                [40, 41, 42, 43, 44],
                                                                                [45, 46, 47, 48, 49],
                                                                                [50, 51, 52, 53, 54],
                                                                                [55, 56, 57, 58, 59],
                                                                                [60, 61, 62, 63, 64],
                                                                                [65, 66, 67, 68, 69]],
                                                                               [[70, 71, 72, 73, 74],
                                                                                [75, 76, 77, 78, 79],
                                                                                [80, 81, 82, 83, 84],
                                                                                [85, 86, 87, 88, 89],
                                                                                [90, 91, 92, 93, 94],
                                                                                [95, 96, 97, 98, 99],
                                                                                [100, 101, 102, 103, 104]]]

    content0 = awkward1.layout.RegularArray(awkward1.layout.NumpyArray(numpy.arange(70)), 5)
    content1 = awkward1.layout.RegularArray(awkward1.layout.NumpyArray(numpy.arange(70)), 5)
    tags = awkward1.layout.Index8(numpy.array([0, 1, 0, 1], dtype=numpy.int8))
    index = awkward1.layout.Index32(numpy.array([0, 0, 1, 1], dtype=numpy.int32))
    union_array = awkward1.layout.UnionArray8_32(tags, index, [ak1, ak2])

    assert awkward1.to_list(union_array) == [[[0, 1, 2, 3, 4],
                                              [5, 6, 7, 8, 9],
                                              [10, 11, 12, 13, 14],
                                              [15, 16, 17, 18, 19],
                                              [20, 21, 22, 23, 24],
                                              [25, 26, 27, 28, 29],
                                              [30, 31, 32, 33, 34]],
                                             [[0, 1, 2, 3, 4],
                                              [5, 6, 7, 8, 9],
                                              [10, 11, 12, 13, 14],
                                              [15, 16, 17, 18, 19],
                                              [20, 21, 22, 23, 24],
                                              [25, 26, 27, 28, 29],
                                              [30, 31, 32, 33, 34]],
                                             [[35, 36, 37, 38, 39],
                                              [40, 41, 42, 43, 44],
                                              [45, 46, 47, 48, 49],
                                              [50, 51, 52, 53, 54],
                                              [55, 56, 57, 58, 59],
                                              [60, 61, 62, 63, 64],
                                              [65, 66, 67, 68, 69]],
                                             [[35, 36, 37, 38, 39],
                                              [40, 41, 42, 43, 44],
                                              [45, 46, 47, 48, 49],
                                              [50, 51, 52, 53, 54],
                                              [55, 56, 57, 58, 59],
                                              [60, 61, 62, 63, 64],
                                              [65, 66, 67, 68, 69]]]

    assert awkward1.to_list(ak1.merge(ak2, 0)) == [[[ 0,  1,  2,  3,  4],
                                                  [ 5,  6,  7,  8,  9],
                                                  [10, 11, 12, 13, 14],
                                                  [15, 16, 17, 18, 19],
                                                  [20, 21, 22, 23, 24],
                                                  [25, 26, 27, 28, 29],
                                                  [30, 31, 32, 33, 34]],

                                                 [[35, 36, 37, 38, 39],
                                                  [40, 41, 42, 43, 44],
                                                  [45, 46, 47, 48, 49],
                                                  [50, 51, 52, 53, 54],
                                                  [55, 56, 57, 58, 59],
                                                  [60, 61, 62, 63, 64],
                                                  [65, 66, 67, 68, 69]],

                                                 [[ 0,  1,  2,  3,  4],
                                                  [ 5,  6,  7,  8,  9],
                                                  [10, 11, 12, 13, 14],
                                                  [15, 16, 17, 18, 19],
                                                  [20, 21, 22, 23, 24],
                                                  [25, 26, 27, 28, 29],
                                                  [30, 31, 32, 33, 34]],

                                                 [[35, 36, 37, 38, 39],
                                                  [40, 41, 42, 43, 44],
                                                  [45, 46, 47, 48, 49],
                                                  [50, 51, 52, 53, 54],
                                                  [55, 56, 57, 58, 59],
                                                  [60, 61, 62, 63, 64],
                                                  [65, 66, 67, 68, 69]],

                                                 [[70, 71, 72, 73, 74],
                                                  [75, 76, 77, 78, 79],
                                                  [80, 81, 82, 83, 84],
                                                  [85, 86, 87, 88, 89],
                                                  [90, 91, 92, 93, 94],
                                                  [95, 96, 97, 98, 99],
                                                  [100, 101, 102, 103, 104]]]

    assert awkward1.to_list(ak1.merge(ak2, 0)) == awkward1.to_list(awkward1.concatenate([np1, np2], 0))
    assert awkward1.to_list(ak1.merge(ak2, 0)) == awkward1.to_list(numpy.concatenate([np1, np2], 0))

    # assert awkward1.to_list(numpy.concatenate([np1, np2], 1)) == [[[0, 1, 2, 3, 4],
    #                                                                [5, 6, 7, 8, 9],
    #                                                                [10, 11, 12, 13, 14],
    #                                                                [15, 16, 17, 18, 19],
    #                                                                [20, 21, 22, 23, 24],
    #                                                                [25, 26, 27, 28, 29],
    #                                                                [30, 31, 32, 33, 34],
    #                                                                [0, 1, 2, 3, 4],
    #                                                                [5, 6, 7, 8, 9],
    #                                                                [10, 11, 12, 13, 14],
    #                                                                [15, 16, 17, 18, 19],
    #                                                                [20, 21, 22, 23, 24],
    #                                                                [25, 26, 27, 28, 29],
    #                                                                [30, 31, 32, 33, 34]],
    #
    #                                                               [[35, 36, 37, 38, 39],
    #                                                                [40, 41, 42, 43, 44],
    #                                                                [45, 46, 47, 48, 49],
    #                                                                [50, 51, 52, 53, 54],
    #                                                                [55, 56, 57, 58, 59],
    #                                                                [60, 61, 62, 63, 64],
    #                                                                [65, 66, 67, 68, 69],
    #                                                                [35, 36, 37, 38, 39],
    #                                                                [40, 41, 42, 43, 44],
    #                                                                [45, 46, 47, 48, 49],
    #                                                                [50, 51, 52, 53, 54],
    #                                                                [55, 56, 57, 58, 59],
    #                                                                [60, 61, 62, 63, 64],
    #                                                                [65, 66, 67, 68, 69]]]
# args = ([array([[[ 0,  1,  2,  3,  4],
#         [ 5,  6,  7,  8,  9],
#         [10, 11, 12, 13, 14],
#         [15, 16, 17, 18, 1...,  89],
#         [ 90,  91,  92,  93,  94],
#         [ 95,  96,  97,  98,  99],
#         [100, 101, 102, 103, 104]]])], 1)
# kwargs = {}
# relevant_args = [array([[[ 0,  1,  2,  3,  4],
#         [ 5,  6,  7,  8,  9],
#         [10, 11, 12, 13, 14],
#         [15, 16, 17, 18, 19...  88,  89],
#         [ 90,  91,  92,  93,  94],
#         [ 95,  96,  97,  98,  99],
#         [100, 101, 102, 103, 104]]])]
#
# >   ???
# E   ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 2 and the array at index 1 has size 3
#
# <__array_function__ internals>:5: ValueError

def test_numpyarray_merge_axis1():

    np1 = numpy.arange(2*7*5).reshape(2, 7, 5)
    np2 = numpy.arange(3*7*5).reshape(3, 7, 5)
    ak1 = awkward1.layout.NumpyArray(np1)
    ak2 = awkward1.layout.NumpyArray(np2)

    assert awkward1.to_list(ak1.merge(ak2, 1)) == [[[0, 1, 2, 3, 4],
                                                   [5, 6, 7, 8, 9],
                                                   [10, 11, 12, 13, 14],
                                                   [15, 16, 17, 18, 19],
                                                   [20, 21, 22, 23, 24],
                                                   [25, 26, 27, 28, 29],
                                                   [30, 31, 32, 33, 34],
                                                   [0, 1, 2, 3, 4],
                                                   [5, 6, 7, 8, 9],
                                                   [10, 11, 12, 13, 14],
                                                   [15, 16, 17, 18, 19],
                                                   [20, 21, 22, 23, 24],
                                                   [25, 26, 27, 28, 29],
                                                   [30, 31, 32, 33, 34]],

                                                  [[35, 36, 37, 38, 39],
                                                   [40, 41, 42, 43, 44],
                                                   [45, 46, 47, 48, 49],
                                                   [50, 51, 52, 53, 54],
                                                   [55, 56, 57, 58, 59],
                                                   [60, 61, 62, 63, 64],
                                                   [65, 66, 67, 68, 69],
                                                   [35, 36, 37, 38, 39],
                                                   [40, 41, 42, 43, 44],
                                                   [45, 46, 47, 48, 49],
                                                   [50, 51, 52, 53, 54],
                                                   [55, 56, 57, 58, 59],
                                                   [60, 61, 62, 63, 64],
                                                   [65, 66, 67, 68, 69]],

                                                   [[70, 71, 72, 73, 74],
                                                   [75, 76, 77, 78, 79],
                                                   [80, 81, 82, 83, 84],
                                                   [85, 86, 87, 88, 89],
                                                   [90, 91, 92, 93, 94],
                                                   [95, 96, 97, 98, 99],
                                                   [100, 101, 102, 103, 104]]]

def test_numpyarray_same_merge_axis1():

    np1 = numpy.arange(2*7*5).reshape(2, 7, 5)
    np2 = numpy.arange(2*7*5, dtype=numpy.int64).reshape(2, 7, 5)
    ak1 = awkward1.layout.NumpyArray(np1)
    ak2 = awkward1.layout.NumpyArray(np2)

    assert awkward1.to_list(ak1.merge(ak2, 1)) == [[[0, 1, 2, 3, 4],
                                                    [5, 6, 7, 8, 9],
                                                    [10, 11, 12, 13, 14],
                                                    [15, 16, 17, 18, 19],
                                                    [20, 21, 22, 23, 24],
                                                    [25, 26, 27, 28, 29],
                                                    [30, 31, 32, 33, 34],
                                                    [0, 1, 2, 3, 4],
                                                    [5, 6, 7, 8, 9],
                                                    [10, 11, 12, 13, 14],
                                                    [15, 16, 17, 18, 19],
                                                    [20, 21, 22, 23, 24],
                                                    [25, 26, 27, 28, 29],
                                                    [30, 31, 32, 33, 34]],

                                                   [[35, 36, 37, 38, 39],
                                                    [40, 41, 42, 43, 44],
                                                    [45, 46, 47, 48, 49],
                                                    [50, 51, 52, 53, 54],
                                                    [55, 56, 57, 58, 59],
                                                    [60, 61, 62, 63, 64],
                                                    [65, 66, 67, 68, 69],
                                                    [35, 36, 37, 38, 39],
                                                    [40, 41, 42, 43, 44],
                                                    [45, 46, 47, 48, 49],
                                                    [50, 51, 52, 53, 54],
                                                    [55, 56, 57, 58, 59],
                                                    [60, 61, 62, 63, 64],
                                                    [65, 66, 67, 68, 69]]]


def test_numpyarray_reverse_merge_axis1():

    np1 = numpy.arange(2*7*5).reshape(2, 7, 5)
    np2 = numpy.arange(3*7*5).reshape(3, 7, 5)
    ak1 = awkward1.layout.NumpyArray(np1)
    ak2 = awkward1.layout.NumpyArray(np2)

    assert awkward1.to_list(ak2.merge(ak1, 1)) == [[[0, 1, 2, 3, 4],
                                                   [5, 6, 7, 8, 9],
                                                   [10, 11, 12, 13, 14],
                                                   [15, 16, 17, 18, 19],
                                                   [20, 21, 22, 23, 24],
                                                   [25, 26, 27, 28, 29],
                                                   [30, 31, 32, 33, 34],
                                                   [0, 1, 2, 3, 4],
                                                   [5, 6, 7, 8, 9],
                                                   [10, 11, 12, 13, 14],
                                                   [15, 16, 17, 18, 19],
                                                   [20, 21, 22, 23, 24],
                                                   [25, 26, 27, 28, 29],
                                                   [30, 31, 32, 33, 34]],

                                                  [[35, 36, 37, 38, 39],
                                                   [40, 41, 42, 43, 44],
                                                   [45, 46, 47, 48, 49],
                                                   [50, 51, 52, 53, 54],
                                                   [55, 56, 57, 58, 59],
                                                   [60, 61, 62, 63, 64],
                                                   [65, 66, 67, 68, 69],
                                                   [35, 36, 37, 38, 39],
                                                   [40, 41, 42, 43, 44],
                                                   [45, 46, 47, 48, 49],
                                                   [50, 51, 52, 53, 54],
                                                   [55, 56, 57, 58, 59],
                                                   [60, 61, 62, 63, 64],
                                                   [65, 66, 67, 68, 69]],

                                                   [[70, 71, 72, 73, 74],
                                                   [75, 76, 77, 78, 79],
                                                   [80, 81, 82, 83, 84],
                                                   [85, 86, 87, 88, 89],
                                                   [90, 91, 92, 93, 94],
                                                   [95, 96, 97, 98, 99],
                                                   [100, 101, 102, 103, 104]]]

def test_numpyarray_same_merge_axis2():

    np1 = numpy.arange(2*7*5).reshape(2, 7, 5)
    np2 = numpy.arange(2*7*5).reshape(2, 7, 5)
    ak1 = awkward1.layout.NumpyArray(np1)
    ak2 = awkward1.layout.NumpyArray(np2)

    assert awkward1.to_list(numpy.concatenate([np1, np2], 2)) == [[[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                                                                   [5, 6, 7, 8, 9, 5, 6, 7, 8, 9],
                                                                   [10, 11, 12, 13, 14, 10, 11, 12, 13, 14],
                                                                   [15, 16, 17, 18, 19, 15, 16, 17, 18, 19],
                                                                   [20, 21, 22, 23, 24, 20, 21, 22, 23, 24],
                                                                   [25, 26, 27, 28, 29, 25, 26, 27, 28, 29],
                                                                   [30, 31, 32, 33, 34, 30, 31, 32, 33, 34]],

                                                                  [[35, 36, 37, 38, 39, 35, 36, 37, 38, 39],
                                                                   [40, 41, 42, 43, 44, 40, 41, 42, 43, 44],
                                                                   [45, 46, 47, 48, 49, 45, 46, 47, 48, 49],
                                                                   [50, 51, 52, 53, 54, 50, 51, 52, 53, 54],
                                                                   [55, 56, 57, 58, 59, 55, 56, 57, 58, 59],
                                                                   [60, 61, 62, 63, 64, 60, 61, 62, 63, 64],
                                                                   [65, 66, 67, 68, 69, 65, 66, 67, 68, 69]]]

    # assert awkward1.to_list(ak1.merge(ak2, 2)) == [[[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
    #                                                 [5, 6, 7, 8, 9, 5, 6, 7, 8, 9],
    #                                                 [10, 11, 12, 13, 14, 10, 11, 12, 13, 14],
    #                                                 [15, 16, 17, 18, 19, 15, 16, 17, 18, 19],
    #                                                 [20, 21, 22, 23, 24, 20, 21, 22, 23, 24],
    #                                                 [25, 26, 27, 28, 29, 25, 26, 27, 28, 29],
    #                                                 [30, 31, 32, 33, 34, 30, 31, 32, 33, 34]],
    #
    #                                                [[35, 36, 37, 38, 39, 35, 36, 37, 38, 39],
    #                                                 [40, 41, 42, 43, 44, 40, 41, 42, 43, 44],
    #                                                 [45, 46, 47, 48, 49, 45, 46, 47, 48, 49],
    #                                                 [50, 51, 52, 53, 54, 50, 51, 52, 53, 54],
    #                                                 [55, 56, 57, 58, 59, 55, 56, 57, 58, 59],
    #                                                 [60, 61, 62, 63, 64, 60, 61, 62, 63, 64],
    #                                                 [65, 66, 67, 68, 69, 65, 66, 67, 68, 69]]]

# def test_numpyarray_merge_axis2():
#
#     np1 = numpy.arange(2*7*5).reshape(2, 7, 5)
#     np2 = numpy.arange(3*7*5).reshape(3, 7, 5)
#     ak1 = awkward1.layout.NumpyArray(np1)
#     ak2 = awkward1.layout.NumpyArray(np2)
#
#     assert awkward1.to_list(ak1.merge(ak2, 2)) == [[[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
#                                                     [5, 6, 7, 8, 9, 5, 6, 7, 8, 9],
#                                                     [10, 11, 12, 13, 14, 10, 11, 12, 13, 14],
#                                                     [15, 16, 17, 18, 19, 15, 16, 17, 18, 19],
#                                                     [20, 21, 22, 23, 24, 20, 21, 22, 23, 24],
#                                                     [25, 26, 27, 28, 29, 25, 26, 27, 28, 29],
#                                                     [30, 31, 32, 33, 34, 30, 31, 32, 33, 34]],
#
#                                                    [[35, 36, 37, 38, 39, 35, 36, 37, 38, 39],
#                                                     [40, 41, 42, 43, 44, 40, 41, 42, 43, 44],
#                                                     [45, 46, 47, 48, 49, 45, 46, 47, 48, 49],
#                                                     [50, 51, 52, 53, 54, 50, 51, 52, 53, 54],
#                                                     [55, 56, 57, 58, 59, 55, 56, 57, 58, 59],
#                                                     [60, 61, 62, 63, 64, 60, 61, 62, 63, 64],
#                                                     [65, 66, 67, 68, 69, 65, 66, 67, 68, 69]],
#
#                                                    [[70, 71, 72, 73, 74],
#                                                     [75, 76, 77, 78, 79],
#                                                     [80, 81, 82, 83, 84],
#                                                     [85, 86, 87, 88, 89],
#                                                     [90, 91, 92, 93, 94],
#                                                     [95, 96, 97, 98, 99],
#                                                     [100, 101, 102, 103, 104]]]
#
# def test_numpyarray_reverse_merge_axis2():
#
#     np1 = numpy.arange(2*7*5).reshape(2, 7, 5)
#     np2 = numpy.arange(3*7*5).reshape(3, 7, 5)
#     ak1 = awkward1.layout.NumpyArray(np1)
#     ak2 = awkward1.layout.NumpyArray(np2)
#
#     assert awkward1.to_list(ak2.merge(ak1, 2)) == [[[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
#                                                     [5, 6, 7, 8, 9, 5, 6, 7, 8, 9],
#                                                     [10, 11, 12, 13, 14, 10, 11, 12, 13, 14],
#                                                     [15, 16, 17, 18, 19, 15, 16, 17, 18, 19],
#                                                     [20, 21, 22, 23, 24, 20, 21, 22, 23, 24],
#                                                     [25, 26, 27, 28, 29, 25, 26, 27, 28, 29],
#                                                     [30, 31, 32, 33, 34, 30, 31, 32, 33, 34]],
#
#                                                    [[35, 36, 37, 38, 39, 35, 36, 37, 38, 39],
#                                                     [40, 41, 42, 43, 44, 40, 41, 42, 43, 44],
#                                                     [45, 46, 47, 48, 49, 45, 46, 47, 48, 49],
#                                                     [50, 51, 52, 53, 54, 50, 51, 52, 53, 54],
#                                                     [55, 56, 57, 58, 59, 55, 56, 57, 58, 59],
#                                                     [60, 61, 62, 63, 64, 60, 61, 62, 63, 64],
#                                                     [65, 66, 67, 68, 69, 65, 66, 67, 68, 69]],
#
#                                                    [[70, 71, 72, 73, 74],
#                                                     [75, 76, 77, 78, 79],
#                                                     [80, 81, 82, 83, 84],
#                                                     [85, 86, 87, 88, 89],
#                                                     [90, 91, 92, 93, 94],
#                                                     [95, 96, 97, 98, 99],
#                                                     [100, 101, 102, 103, 104]]]


    # awkward1.to_list(numpy.concatenate([np1, np2], 2))
                                    # [[[ 0,  1,  2,  3,  4],
                                    #   [ 5,  6,  7,  8,  9],
                                    #   [10, 11, 12, 13, 14],
                                    #   [15, 16, 17, 18, 19],
                                    #   [20, 21, 22, 23, 24],
                                    #   [25, 26, 27, 28, 29],
                                    #   [30, 31, 32, 33, 34],
                                    #   ],
                                    #
                                    #  [[35, 36, 37, 38, 39],
                                    #   [40, 41, 42, 43, 44],
                                    #   [45, 46, 47, 48, 49],
                                    #   [50, 51, 52, 53, 54],
                                    #   [55, 56, 57, 58, 59],
                                    #   [60, 61, 62, 63, 64],
                                    #   [65, 66, 67, 68, 69]]]

    # assert awkward1.to_list(numpy.concatenate([np1, np2], 2)) == []
    # awkward1.to_list(numpy.concatenate([np1, np2], 1))
    # assert awkward1.to_list(ak1[1:, :-1, ::-1].merge(ak2[1:, :-1, ::-1])) == awkward1.to_list(numpy.concatenate([np1[1:, :-1, ::-1], np2[1:, :-1, ::-1]]))

    # for x in [numpy.bool, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float32, numpy.float64]:
    #     for y in [numpy.bool, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float32, numpy.float64]:
    #         z = numpy.concatenate([numpy.array([1, 2, 3], dtype=x), numpy.array([4, 5], dtype=y)]).dtype.type
    #         one = awkward1.layout.NumpyArray(numpy.array([1, 2, 3], dtype=x))
    #         two = awkward1.layout.NumpyArray(numpy.array([4, 5], dtype=y))
    #         three = one.merge(two)
    #         assert numpy.asarray(three).dtype == numpy.dtype(z), "{0} {1} {2} {3}".format(x, y, z, numpy.asarray(three).dtype.type)
    #         assert awkward1.to_list(three) == awkward1.to_list(numpy.concatenate([numpy.asarray(one), numpy.asarray(two)]))
    #         assert awkward1.to_list(one.merge(emptyarray)) == awkward1.to_list(one)
    #         assert awkward1.to_list(emptyarray.merge(one)) == awkward1.to_list(one)
