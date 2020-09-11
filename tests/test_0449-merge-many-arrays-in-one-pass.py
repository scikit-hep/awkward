# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test_numpyarray():
    for dtype1 in ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "?"):
        for dtype2 in ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "?"):
            for dtype3 in ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "?"):
                for dtype4 in ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "?"):
                    one   = numpy.array([0, 1, 2], dtype=dtype1)
                    two   = numpy.array([3, 0], dtype=dtype2)
                    three = numpy.array([], dtype=dtype3)
                    four  = numpy.array([4, 5, 0, 6, 7], dtype=dtype4)
                    combined = numpy.concatenate([one, two, three, four])

                    ak_combined = awkward1.layout.NumpyArray(one).mergemany([
                        awkward1.layout.NumpyArray(two),
                        awkward1.layout.NumpyArray(three),
                        awkward1.layout.NumpyArray(four),
                    ])

                    assert awkward1.to_list(ak_combined) == combined.tolist()
                    assert awkward1.to_numpy(ak_combined).dtype == combined.dtype

                    ak_combined = awkward1.layout.NumpyArray(one).mergemany([
                        awkward1.layout.NumpyArray(two),
                        awkward1.layout.EmptyArray(),
                        awkward1.layout.NumpyArray(four),
                    ])

                    assert awkward1.to_list(ak_combined) == combined.tolist()
                    assert awkward1.to_numpy(ak_combined).dtype == numpy.concatenate([one, two, four]).dtype

def test_lists():
    one = awkward1.Array([[1, 2, 3], [], [4, 5]]).layout
    two = awkward1.Array([[1.1, 2.2], [3.3, 4.4]]).layout
    three = awkward1.layout.EmptyArray()
    four = awkward1.from_numpy(numpy.array([[10], [20]]), regulararray=True, highlevel=False)
    assert awkward1.to_list(one.mergemany([two, three, four])) == [[1.0, 2.0, 3.0], [], [4.0, 5.0], [1.1, 2.2], [3.3, 4.4], [10.0], [20.0]]
    assert awkward1.to_list(four.mergemany([three, two, one])) == [[10.0], [20.0], [1.1, 2.2], [3.3, 4.4], [1.0, 2.0, 3.0], [], [4.0, 5.0]]

    one = awkward1.layout.ListArray64(one.starts, one.stops, one.content)
    two = awkward1.layout.ListArray64(two.starts, two.stops, two.content)
    assert awkward1.to_list(one.mergemany([two, three, four])) == [[1.0, 2.0, 3.0], [], [4.0, 5.0], [1.1, 2.2], [3.3, 4.4], [10.0], [20.0]]
    assert awkward1.to_list(four.mergemany([three, two, one])) == [[10.0], [20.0], [1.1, 2.2], [3.3, 4.4], [1.0, 2.0, 3.0], [], [4.0, 5.0]]

def test_records():
    one = awkward1.Array([{"x": 1, "y": [1]}, {"x": 2, "y": [1, 2]}, {"x": 3, "y": [1, 2, 3]}]).layout
    two = awkward1.Array([{"y": [], "x": 4}, {"y": [3, 2, 1], "x": 5}]).layout
    three = two[0:0]
    four = awkward1.Array([{"x": 6, "y": [1]}, {"x": 7, "y": [1, 2]}]).layout
    assert awkward1.to_list(one.mergemany([two, three, four])) == [{"x": 1, "y": [1]}, {"x": 2, "y": [1, 2]}, {"x": 3, "y": [1, 2, 3]}, {"y": [], "x": 4}, {"y": [3, 2, 1], "x": 5}, {"x": 6, "y": [1]}, {"x": 7, "y": [1, 2]}]

    three = awkward1.layout.EmptyArray()
    assert awkward1.to_list(one.mergemany([two, three, four])) == [{"x": 1, "y": [1]}, {"x": 2, "y": [1, 2]}, {"x": 3, "y": [1, 2, 3]}, {"y": [], "x": 4}, {"y": [3, 2, 1], "x": 5}, {"x": 6, "y": [1]}, {"x": 7, "y": [1, 2]}]

def test_tuples():
    one = awkward1.Array([(1, [1]), (2, [1, 2]), (3, [1, 2, 3])]).layout
    two = awkward1.Array([(4, []), (5, [3, 2, 1])]).layout
    three = two[0:0]
    four = awkward1.Array([(6, [1]), (7, [1, 2])]).layout
    assert awkward1.to_list(one.mergemany([two, three, four])) == [(1, [1]), (2, [1, 2]), (3, [1, 2, 3]), (4, []), (5, [3, 2, 1]), (6, [1]), (7, [1, 2])]

    three = awkward1.layout.EmptyArray()
    assert awkward1.to_list(one.mergemany([two, three, four])) == [(1, [1]), (2, [1, 2]), (3, [1, 2, 3]), (4, []), (5, [3, 2, 1]), (6, [1]), (7, [1, 2])]

def test_indexed():
    one = awkward1.Array([1, 2, 3, None, 4, None, None, 5]).layout
    two = awkward1.Array([6, 7, 8]).layout
    three = awkward1.layout.EmptyArray()
    four = awkward1.Array([9, None, None]).layout
    assert awkward1.to_list(one.mergemany([two, three, four])) == [1, 2, 3, None, 4, None, None, 5, 6, 7, 8, 9, None, None]

def test_reverse_indexed():
    one = awkward1.Array([1, 2, 3]).layout
    two = awkward1.Array([4, 5]).layout
    three = awkward1.Array([None, 6, None]).layout
    assert awkward1.to_list(one.mergemany([two, three])) == [1, 2, 3, 4, 5, None, 2, None]

    four = awkward1.Array([7, 8, None, None, 9]).layout
    assert awkward1.to_list(one.mergemany([two, three, four])) == [1, 2, 3, 4, 5, None, 2, None, 7, 8, None, None, 9]
