# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1 as ak

def test_list_offset_array_concatenate():
    content_one = ak.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    content_two = ak.layout.NumpyArray(numpy.array([999.999, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99]))
    offsets_one = ak.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10], dtype=numpy.int64))
    offsets_two = ak.layout.Index64(numpy.array([1, 3, 4, 4, 6, 9, 9, 10], dtype=numpy.int64))
    offsets_three = ak.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64))
    offsets_four = ak.layout.Index64(numpy.array([1, 3, 4, 4, 6, 7, 7], dtype=numpy.int64))

    one = ak.layout.ListOffsetArray64(offsets_one, content_one)
    padded_one = one.rpad(7, 0)
    assert ak.to_list(padded_one) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9], None, None]

    two = ak.layout.ListOffsetArray64(offsets_two, content_two)
    three = ak.layout.ListOffsetArray64(offsets_three, one)
    four = ak.layout.ListOffsetArray64(offsets_four, two)

    assert ak.to_list(ak.concatenate([one, two], 0)) == [
        [0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9],
        [0.11, 0.22], [0.33], [], [0.44, 0.55], [0.66, 0.77, 0.88], [], [0.99]]

    assert ak.to_list(ak.concatenate([padded_one, two], 1)) == [
        [0.0, 1.1, 2.2, 0.11, 0.22],
        [0.33],
        [3.3, 4.4],
        [5.5, 0.44, 0.55],
        [6.6, 7.7, 8.8, 9.9, 0.66, 0.77, 0.88],
        [],
        [0.99]]

    with pytest.raises(ValueError) as err:
        assert ak.to_list(ak.concatenate([one, two], 2))

    assert ak.to_list(ak.concatenate([three, four], 0)) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
         [],
        [[5.5], [6.6, 7.7, 8.8, 9.9]],
        [[0.33], []],
        [[0.44, 0.55]],
         [],
        [[0.66, 0.77, 0.88], []],
        [[0.99]],
         []]

    padded = three.rpad(6, 0)
    assert ak.to_list(ak.concatenate([padded, four], 1)) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [0.33], []],
        [[0.44, 0.55]],
        [[5.5], [6.6, 7.7, 8.8, 9.9]],
        [[0.66, 0.77, 0.88], []],
        [[0.99]],
        []]

    assert ak.to_list(ak.concatenate([four, four], 2)) == [
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

    with pytest.raises(ValueError) as err:
        ak.to_list(ak.concatenate([one, two], 2))

def test_indexed_array_concatenate():
    one = ak.Array([[1, 2, 3], [None, 4], None, [None, 5]]).layout
    two = ak.Array([6, 7, 8]).layout
    three = ak.Array([[6.6], [7.7, 8.8]]).layout
    four = ak.Array([[6.6], [7.7, 8.8], None, [9.9]]).layout

    assert ak.to_list(ak.concatenate([one, two], 0)) == [[1, 2, 3], [None, 4], None, [None, 5], 6, 7, 8]

    with pytest.raises(ValueError) as err:
        ak.to_list(ak.concatenate([one, three], 1))

    assert ak.to_list(ak.concatenate([one, four], 1)) == [[1, 2, 3, 6.6], [None, 4, 7.7, 8.8], [], [None, 5, 9.9]]

def test_bytemasked_concatenate():
    one = ak.Array([1, 2, 3, 4, 5, 6]).mask[[True, True, False, True, False, True]].layout
    two = ak.Array([7, 99, 999, 8, 9]).mask[[True, False, False, True, True]].layout

    assert ak.to_list(ak.concatenate([one, two], 0)) == [1, 2, None, 4, None, 6, 7, None, None, 8, 9]

    with pytest.raises(ValueError) as err:
        ak.to_list(ak.concatenate([one, two], 1))

def test_listoffsetarray_concatenate():
    content_one = ak.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    offsets_one = ak.layout.Index64(numpy.array([0, 3, 3, 5, 9]))
    one = ak.layout.ListOffsetArray64(offsets_one, content_one)

    assert ak.to_list(one) == [[1, 2, 3], [], [4, 5], [6, 7, 8, 9]]

    content_two = ak.layout.NumpyArray(numpy.array([100, 200, 300, 400, 500]))
    offsets_two = ak.layout.Index64(numpy.array([0, 2, 4, 4, 5]))
    two = ak.layout.ListOffsetArray64(offsets_two, content_two)

    assert ak.to_list(two) == [[100, 200], [300, 400], [], [500]]
    assert ak.to_list(ak.concatenate([one, two], 0)) == [[1, 2, 3], [], [4, 5], [6, 7, 8, 9], [100, 200], [300, 400], [], [500]]
    assert ak.to_list(ak.concatenate([one, two], 1)) == [[1, 2, 3, 100, 200], [300, 400], [4, 5], [6, 7, 8, 9, 500]]

def test_numpyarray_concatenate_axis0():
    emptyarray = ak.layout.EmptyArray()

    np1 = numpy.arange(2*7*5, dtype=numpy.float64).reshape(2, 7, 5)
    np2 = numpy.arange(3*7*5, dtype=numpy.int64).reshape(3, 7, 5)
    ak1 = ak.layout.NumpyArray(np1)
    ak2 = ak.layout.NumpyArray(np2)

    assert ak.to_list(ak.concatenate([np1, np2, np1, np2], 0)) == ak.to_list(numpy.concatenate([np1, np2, np1, np2], 0))
    assert ak.to_list(ak.concatenate([ak1, ak2], 0)) == ak.to_list(numpy.concatenate([ak1, ak2], 0))
    assert ak.to_list(numpy.concatenate([ak1, ak2], 0)) == ak.to_list(ak.concatenate([np1, np2], 0))

def test_numpyarray_concatenate():

    np1 = numpy.arange(2*7*5, dtype=numpy.float64).reshape(2, 7, 5)
    np2 = numpy.arange(2*7*5, dtype=numpy.int64).reshape(2, 7, 5)
    ak1 = ak.layout.NumpyArray(np1)
    ak2 = ak.layout.NumpyArray(np2)

    assert ak.to_list(numpy.concatenate([np1, np2], 1)) == ak.to_list(ak.concatenate([ak1, ak2], 1))
    assert ak.to_list(numpy.concatenate([np2, np1], 1)) == ak.to_list(ak.concatenate([ak2, ak1], 1))
    assert ak.to_list(numpy.concatenate([np1, np2], 2)) == ak.to_list(ak.concatenate([ak1, ak2], 2))
    assert ak.to_list(numpy.concatenate([np2, np1], 2)) == ak.to_list(ak.concatenate([ak2, ak1], 2))

def test_numbers_and_records_concatenate():
    numbers = [
            ak.Array([
                [1.1, 2.2, 3.3],
                [],
                [4.4, 5.5],
                [6.6, 7.7, 8.8, 9.9],
            ]),
            ak.Array([
                [10, 20],
                [30],
                [],
                [40, 50, 60],
            ]),
            ak.Array(numpy.array([
                [101, 102, 103],
                [201, 202, 203],
                [301, 302, 303],
                [401, 402, 403],
            ])),
        ]

    records = [
        ak.Array([
            [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}],
            [],
            [{"x": 2.2, "y": [1, 2]}],
            [{"x": 3.3, "y": [1, 2, 3]}, {"x": 4.4, "y": [1, 2, 3, 4]}],
        ]),
        ak.Array([
            [{"x": 0.0, "y": []}],
            [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}],
            [],
            [{"x": 3.3, "y": [1, 2, 3]}],
        ]),
    ]

    assert ak.concatenate([numbers[0], records[0]], 1) == [
        [1.1, 2.2, 3.3, {"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}],
        [],
        [4.4, 5.5, {"x": 2.2, "y": [1, 2]}],
        [6.6, 7.7, 8.8, 9.9, {"x": 3.3, "y": [1, 2, 3]}, {"x": 4.4, "y": [1, 2, 3, 4]}],
    ]

    numbers = [
        ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]),
        ak.Array([[10, 20], [30], [], [40, 50, 60]]),
        ak.Array(numpy.array([[101, 102, 103], [201, 202, 203], [301, 302, 303], [401, 402, 403]])),
    ]
    records = [
        ak.Array([[{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}], [], [{"x": 2.2, "y": [1, 2]}]]),
        ak.Array([[{"x": 3.3, "y": [1, 2, 3]}], [{"x": 4.4, "y": [2, 1]}, {"x": 5.5, "y": [1]}], []]),
    ]

    assert ak.concatenate([numbers[0], numbers[1]], 1) == [
        [1.1, 2.2, 3.3, 10, 20],
        [30],
        [4.4, 5.5],
        [6.6, 7.7, 8.8, 9.9, 40, 50, 60],
    ]

    assert ak.to_list(ak.concatenate(numbers, 1)) == [
        [1.1, 2.2, 3.3, 10, 20, 101, 102, 103],
        [30, 201, 202, 203],
        [4.4, 5.5, 301, 302, 303],
        [6.6, 7.7, 8.8, 9.9, 40, 50, 60, 401, 402, 403]]

    assert ak.to_list(ak.concatenate(records, 1)) == [
        [{'x': 0.0, 'y': []}, {'x': 1.1, 'y': [1]}, {'x': 3.3, 'y': [1, 2, 3]}],
        [{'x': 4.4, 'y': [2, 1]}, {'x': 5.5, 'y': [1]}],
        [{'x': 2.2, 'y': [1, 2]}]]

    with pytest.raises(ValueError) as err:
        ak.to_list(ak.concatenate([numbers, records], 1))
    assert str(err.value).startswith("scalar Record can only be sliced by field name (string); try \"0\"")

def test_jims_test():
    arrays = [
    ak.Array([[[0.0, 1.1, 2.2], []], [[3.3, 4.4]], [[5.5], [6.6, 7.7, 8.8, 9.9]]]),
    ak.Array([[[10, 20], [30]], [[40]], [[50, 60, 70], [80, 90]]]),
    ]
    # nothing is required to have the same length
    assert ak.concatenate(arrays, axis=0).tolist() == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4]],
        [[5.5], [6.6, 7.7, 8.8, 9.9]],
        [[10, 20], [30]],
        [[40]],
        [[50, 60, 70], [80, 90]],
    ]
    # the outermost arrays are required to have the same length, but nothing deeper than that
    assert ak.concatenate(arrays, axis=1).tolist() == [
        [[0.0, 1.1, 2.2], [], [10, 20], [30]],
        [[3.3, 4.4], [40]],
        [[5.5], [6.6, 7.7, 8.8, 9.9], [50, 60, 70], [80, 90]],
    ]
    # the outermost arrays and the first level are required to have the same length, but nothing deeper
    assert ak.concatenate(arrays, axis=2).tolist() == [
        [[0.0, 1.1, 2.2, 10, 20], [30]],
        [[3.3, 4.4, 40]],
        [[5.5, 50, 60, 70], [6.6, 7.7, 8.8, 9.9, 80, 90]],
    ]
