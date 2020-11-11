# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1 as ak

def test_list_offset_array_concatenate():
    content = ak.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    content2 = ak.layout.NumpyArray(numpy.array([999.999, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99]))
    offsets = ak.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10], dtype=numpy.int64))
    array   = ak.layout.ListOffsetArray64(offsets, content)
    padded = array.rpad(7, 0)
    assert ak.to_list(padded) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9], None, None]

    # array1 = [[[1.1 2.2] [3.3]] [] [[4.4 5.5]] [[6.6 7.7 8.8] [] [9.9]]]
    # array2 = [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]
    offsets2 = ak.layout.Index64(numpy.array([1, 3, 4, 4, 6, 9, 9, 10], dtype=numpy.int64))
    array2   = ak.layout.ListOffsetArray64(offsets2, content)
    array22   = ak.layout.ListOffsetArray64(offsets2, content2)

    assert ak.to_list(array2) == [[1.1, 2.2], [3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8], [], [9.9]]
    assert ak.to_list(ak.concatenate([padded, array2], 0)) == [[0.0, 1.1, 2.2],
                                                        [],
                                                        [3.3, 4.4],
                                                        [5.5],
                                                        [6.6, 7.7, 8.8, 9.9], None, None,
                                                        [1.1, 2.2],
                                                        [3.3],
                                                        [],
                                                        [4.4, 5.5],
                                                        [6.6, 7.7, 8.8],
                                                        [],
                                                        [9.9]]

    assert ak.to_list(ak.concatenate([padded, array2], 1)) == [[0.0, 1.1, 2.2, 1.1, 2.2],
     [3.3],
     [3.3, 4.4],
     [5.5, 4.4, 5.5],
     [6.6, 7.7, 8.8, 9.9, 6.6, 7.7, 8.8],
     [],
     [9.9]]

    assert ak.to_list(padded) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9], None, None]

    assert ak.to_list(ak.concatenate([padded, array2], 1)) == [[0.0, 1.1, 2.2, 1.1, 2.2],
     [3.3],
     [3.3, 4.4],
     [5.5, 4.4, 5.5],
     [6.6, 7.7, 8.8, 9.9, 6.6, 7.7, 8.8],
     [],
     [9.9]]

    with pytest.raises(ValueError) as err:
        assert ak.to_list(ak.concatenate([array, array2], 2))
    assert str(err.value).startswith("all arrays must have the same length for concatenate in axis > 0")

    offsets3 = ak.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64))
    array3   = ak.layout.ListOffsetArray64(offsets3, array)

    assert ak.to_list(array3) == [[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5], [6.6, 7.7, 8.8, 9.9]]]

    offsets4 = ak.layout.Index64(numpy.array([1, 3, 4, 4, 6, 7, 7], dtype=numpy.int64))
    array4   = ak.layout.ListOffsetArray64(offsets4, array22)

    assert ak.to_list(array4) == [[[0.33], []], [[0.44, 0.55]], [], [[0.66, 0.77, 0.88], []], [[0.99]], []]

    assert ak.to_list(ak.concatenate([array3, array4], 0)) == [[[0.0, 1.1, 2.2], [], [3.3, 4.4]],
                                                         [],
                                                         [[5.5], [6.6, 7.7, 8.8, 9.9]],
                                                         [[0.33], []],
                                                         [[0.44, 0.55]],
                                                         [],
                                                         [[0.66, 0.77, 0.88], []],
                                                         [[0.99]],
                                                         []]

    padded = array3.rpad(6, 0)
    assert ak.to_list(ak.concatenate([padded, array4], 1)) == [[[0.0, 1.1, 2.2], [], [3.3, 4.4], [0.33], []],
                                                         [[0.44, 0.55]],
                                                         [[5.5], [6.6, 7.7, 8.8, 9.9]],
                                                         [[0.66, 0.77, 0.88], []],
                                                         [[0.99]],
                                                         []]

    assert ak.to_list(ak.concatenate([array4, array4], 2)) == [
        [[0.33, 0.33], []],
        [[0.44, 0.55, 0.44, 0.55]],
        [],
        [[0.66, 0.77, 0.88, 0.66, 0.77, 0.88], []],
        [[0.99, 0.99]],
        []]

def test_list_array_concatenate():
    one = ak.Array([[1, 2, 3], [], [4, 5]]).layout
    two = ak.Array([[1.1, 2.2], [3.3, 4.4], [5.5]]).layout

    one = ak.layout.ListArray64(one.starts, one.stops, one.content)
    two = ak.layout.ListArray64(two.starts, two.stops, two.content)
    assert ak.to_list(ak.concatenate([one, two], 0)) == [[1, 2, 3], [], [4, 5], [1.1, 2.2], [3.3, 4.4], [5.5]]
    assert ak.to_list(ak.concatenate([one, two], 1)) == [[1, 2, 3, 1.1, 2.2], [3.3, 4.4], [4, 5, 5.5]]

def test_records_concatenate():
    one = ak.Array([{"x": 1, "y": [1]}, {"x": 2, "y": [1, 2]}, {"x": 3, "y": [1, 2, 3]}]).layout
    two = ak.Array([{"y": [], "x": 4}, {"y": [3, 2, 1], "x": 5}]).layout
    assert ak.to_list(ak.concatenate([one, two], 0)) == [{"x": 1, "y": [1]}, {"x": 2, "y": [1, 2]}, {"x": 3, "y": [1, 2, 3]},
                                                   {"y": [], "x": 4}, {"y": [3, 2, 1], "x": 5}]
    with pytest.raises(ValueError) as err:
        ak.to_list(ak.concatenate([one, two], 1))
    assert str(err.value).startswith("all arrays must have the same length for concatenate in axis > 0")

    with pytest.raises(ValueError) as err:
        ak.to_list(ak.concatenate([one, two], 2))
    assert str(err.value).startswith("all arrays must have the same length for concatenate in axis > 0")

def test_indexed_array_concatenate():
    one = ak.Array([[1, 2, 3], [None, 4], None, [None, 5]]).layout
    two = ak.Array([6, 7, 8]).layout
    three = ak.Array([[6.6], [7.7, 8.8]]).layout
    four = ak.Array([[6.6], [7.7, 8.8], None, [9.9]]).layout

    assert ak.to_list(ak.concatenate([one, two], 0)) == [[1, 2, 3], [None, 4], None, [None, 5], 6, 7, 8]
    with pytest.raises(ValueError) as err:
        ak.to_list(ak.concatenate([one, three], 1))
    assert str(err.value).startswith("all arrays must have the same length for concatenate in axis > 0")

    assert ak.to_list(ak.concatenate([one, four], 1)) == [[1, 2, 3, 6.6], [None, 4, 7.7, 8.8], [], [None, 5, 9.9]]

def test_bytemasked_concatenate():
    one = ak.Array([1, 2, 3, 4, 5, 6]).mask[[True, True, False, True, False, True]].layout
    two = ak.Array([7, 99, 999, 8, 9]).mask[[True, False, False, True, True]].layout

    assert ak.to_list(ak.concatenate([one, two], 0)) == [1, 2, None, 4, None, 6, 7, None, None, 8, 9]

    with pytest.raises(ValueError) as err:
        ak.to_list(ak.concatenate([one, two], 1))
    assert str(err.value).startswith("all arrays must have the same length for concatenate in axis > 0")

def test_listoffsetarray_concatenate():
    content1 = ak.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    offsets1 = ak.layout.Index64(numpy.array([0, 3, 3, 5, 9]))
    listoffsetarray1 = ak.layout.ListOffsetArray64(offsets1, content1)

    assert ak.to_list(listoffsetarray1) == [[1, 2, 3], [], [4, 5], [6, 7, 8, 9]]

    content2 = ak.layout.NumpyArray(numpy.array([100, 200, 300, 400, 500]))
    offsets2 = ak.layout.Index64(numpy.array([0, 2, 4, 4, 5]))
    listoffsetarray2 = ak.layout.ListOffsetArray64(offsets2, content2)

    assert ak.to_list(listoffsetarray2) == [[100, 200], [300, 400], [], [500]]
    assert ak.to_list(ak.concatenate([listoffsetarray1, listoffsetarray2], 0)) == [[1, 2, 3], [], [4, 5], [6, 7, 8, 9], [100, 200], [300, 400], [], [500]]
    assert ak.to_list(ak.concatenate([listoffsetarray1, listoffsetarray2], 1)) == [[1, 2, 3, 100, 200], [300, 400], [4, 5], [6, 7, 8, 9, 500]]

def test_numpyarray_concatenate():
    emptyarray = ak.layout.EmptyArray()

    np1 = numpy.arange(2*7*5).reshape(2, 7, 5)
    np2 = numpy.arange(3*7*5).reshape(3, 7, 5)
    ak1 = ak.layout.NumpyArray(np1)
    ak2 = ak.layout.NumpyArray(np2)

    assert ak.to_list(ak1) == [[[ 0,  1,  2,  3,  4],
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

    assert ak.to_list(ak2) == [[[ 0,  1,  2,  3,  4],
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

    assert ak.to_list(ak.concatenate([np1, np2, np1, np2], 0)) == [[[0, 1, 2, 3, 4],
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

    content0 = ak.layout.RegularArray(ak.layout.NumpyArray(numpy.arange(70)), 5)
    content1 = ak.layout.RegularArray(ak.layout.NumpyArray(numpy.arange(70)), 5)
    tags = ak.layout.Index8(numpy.array([0, 1, 0, 1], dtype=numpy.int8))
    index = ak.layout.Index32(numpy.array([0, 0, 1, 1], dtype=numpy.int32))
    union_array = ak.layout.UnionArray8_32(tags, index, [ak1, ak2])

    assert ak.to_list(union_array) == [[[0, 1, 2, 3, 4],
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

    assert ak.to_list(ak.concatenate([ak1, ak2], 0)) == [[[ 0,  1,  2,  3,  4],
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

    assert ak.to_list(ak1.merge(ak2, 0)) == ak.to_list(ak.concatenate([np1, np2], 0))

def test_numpyarray_concatenate_axis1():

    np1 = numpy.arange(2*7*5).reshape(2, 7, 5)
    np2 = numpy.arange(2*7*5, dtype=numpy.int64).reshape(2, 7, 5)
    ak1 = ak.layout.NumpyArray(np1)
    ak2 = ak.layout.NumpyArray(np2)

    assert ak.to_list(numpy.concatenate([np1, np2], 1)) == ak.to_list(ak.concatenate([ak1, ak2], 1))

def test_numpyarray_reverse_concatenate_axis1():

    np1 = numpy.arange(2*7*5).reshape(2, 7, 5)
    np2 = numpy.arange(2*7*5).reshape(2, 7, 5)
    ak1 = ak.layout.NumpyArray(np1)
    ak2 = ak.layout.NumpyArray(np2)

    assert ak.to_list(numpy.concatenate([np2, np1], 1)) == ak.to_list(ak.concatenate([ak2, ak1], 1))

def test_numpyarray_concatenate_axis2():

    np1 = numpy.arange(2*7*5).reshape(2, 7, 5)
    np2 = numpy.arange(2*7*5).reshape(2, 7, 5)
    ak1 = ak.layout.NumpyArray(np1)
    ak2 = ak.layout.NumpyArray(np2)

    assert ak.to_list(numpy.concatenate([np1, np2], 2)) == ak.to_list(ak.concatenate([ak1, ak2], 2))

def test_numpyarray_reverse_concatenate_axis2():

    np1 = numpy.arange(2*7*5).reshape(2, 7, 5)
    np2 = numpy.arange(2*7*5).reshape(2, 7, 5)
    ak1 = ak.layout.NumpyArray(np1)
    ak2 = ak.layout.NumpyArray(np2)

    assert ak.to_list(numpy.concatenate([np2, np1], 2)) == ak.to_list(ak.concatenate([ak2, ak1], 2))
