# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak


def test_numpyarray():
    for dtype1 in ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "?"):
        for dtype2 in ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "?"):
            for dtype3 in ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "?"):
                for dtype4 in ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "?"):
                    one   = np.array([0, 1, 2], dtype=dtype1)
                    two   = np.array([3, 0], dtype=dtype2)
                    three = np.array([], dtype=dtype3)
                    four  = np.array([4, 5, 0, 6, 7], dtype=dtype4)
                    combined = np.concatenate([one, two, three, four])

                    ak_combined = ak.layout.NumpyArray(one).mergemany([
                        ak.layout.NumpyArray(two),
                        ak.layout.NumpyArray(three),
                        ak.layout.NumpyArray(four),
                    ])

                    assert ak.to_list(ak_combined) == combined.tolist()
                    assert ak.to_numpy(ak_combined).dtype == combined.dtype

                    ak_combined = ak.layout.NumpyArray(one).mergemany([
                        ak.layout.NumpyArray(two),
                        ak.layout.EmptyArray(),
                        ak.layout.NumpyArray(four),
                    ])

                    assert ak.to_list(ak_combined) == combined.tolist()
                    assert ak.to_numpy(ak_combined).dtype == np.concatenate([one, two, four]).dtype

def test_lists():
    one = ak.Array([[1, 2, 3], [], [4, 5]]).layout
    two = ak.Array([[1.1, 2.2], [3.3, 4.4]]).layout
    three = ak.layout.EmptyArray()
    four = ak.from_numpy(np.array([[10], [20]]), regulararray=True, highlevel=False)
    assert ak.to_list(one.mergemany([two, three, four])) == [[1.0, 2.0, 3.0], [], [4.0, 5.0], [1.1, 2.2], [3.3, 4.4], [10.0], [20.0]]
    assert ak.to_list(four.mergemany([three, two, one])) == [[10.0], [20.0], [1.1, 2.2], [3.3, 4.4], [1.0, 2.0, 3.0], [], [4.0, 5.0]]

    one = ak.layout.ListArray64(one.starts, one.stops, one.content)
    two = ak.layout.ListArray64(two.starts, two.stops, two.content)
    assert ak.to_list(one.mergemany([two, three, four])) == [[1.0, 2.0, 3.0], [], [4.0, 5.0], [1.1, 2.2], [3.3, 4.4], [10.0], [20.0]]
    assert ak.to_list(four.mergemany([three, two, one])) == [[10.0], [20.0], [1.1, 2.2], [3.3, 4.4], [1.0, 2.0, 3.0], [], [4.0, 5.0]]

def test_records():
    one = ak.Array([{"x": 1, "y": [1]}, {"x": 2, "y": [1, 2]}, {"x": 3, "y": [1, 2, 3]}]).layout
    two = ak.Array([{"y": [], "x": 4}, {"y": [3, 2, 1], "x": 5}]).layout
    three = two[0:0]
    four = ak.Array([{"x": 6, "y": [1]}, {"x": 7, "y": [1, 2]}]).layout
    assert ak.to_list(one.mergemany([two, three, four])) == [{"x": 1, "y": [1]}, {"x": 2, "y": [1, 2]}, {"x": 3, "y": [1, 2, 3]}, {"y": [], "x": 4}, {"y": [3, 2, 1], "x": 5}, {"x": 6, "y": [1]}, {"x": 7, "y": [1, 2]}]

    three = ak.layout.EmptyArray()
    assert ak.to_list(one.mergemany([two, three, four])) == [{"x": 1, "y": [1]}, {"x": 2, "y": [1, 2]}, {"x": 3, "y": [1, 2, 3]}, {"y": [], "x": 4}, {"y": [3, 2, 1], "x": 5}, {"x": 6, "y": [1]}, {"x": 7, "y": [1, 2]}]

def test_tuples():
    one = ak.Array([(1, [1]), (2, [1, 2]), (3, [1, 2, 3])]).layout
    two = ak.Array([(4, []), (5, [3, 2, 1])]).layout
    three = two[0:0]
    four = ak.Array([(6, [1]), (7, [1, 2])]).layout
    assert ak.to_list(one.mergemany([two, three, four])) == [(1, [1]), (2, [1, 2]), (3, [1, 2, 3]), (4, []), (5, [3, 2, 1]), (6, [1]), (7, [1, 2])]

    three = ak.layout.EmptyArray()
    assert ak.to_list(one.mergemany([two, three, four])) == [(1, [1]), (2, [1, 2]), (3, [1, 2, 3]), (4, []), (5, [3, 2, 1]), (6, [1]), (7, [1, 2])]

def test_indexed():
    one = ak.Array([1, 2, 3, None, 4, None, None, 5]).layout
    two = ak.Array([6, 7, 8]).layout
    three = ak.layout.EmptyArray()
    four = ak.Array([9, None, None]).layout
    assert ak.to_list(one.mergemany([two, three, four])) == [1, 2, 3, None, 4, None, None, 5, 6, 7, 8, 9, None, None]

def test_reverse_indexed():
    one = ak.Array([1, 2, 3]).layout
    two = ak.Array([4, 5]).layout
    three = ak.Array([None, 6, None]).layout
    assert ak.to_list(one.mergemany([two, three])) == [1, 2, 3, 4, 5, None, 6, None]

    four = ak.Array([7, 8, None, None, 9]).layout
    assert ak.to_list(one.mergemany([two, three, four])) == [1, 2, 3, 4, 5, None, 6, None, 7, 8, None, None, 9]

def test_bytemasked():
    one = ak.Array([1, 2, 3, 4, 5, 6]).mask[[True, True, False, True, False, True]].layout
    two = ak.Array([7, 99, 999, 8, 9]).mask[[True, False, False, True, True]].layout
    three = ak.Array([100, 200, 300]).layout
    four = ak.Array([None, None, 123, None]).layout
    assert ak.to_list(one.mergemany([two, three, four])) == [1, 2, None, 4, None, 6, 7, None, None, 8, 9, 100, 200, 300, None, None, 123, None]
    assert ak.to_list(four.mergemany([three, two, one])) == [None, None, 123, None, 100, 200, 300, 7, None, None, 8, 9, 1, 2, None, 4, None, 6]
    assert ak.to_list(three.mergemany([four, one])) == [100, 200, 300, None, None, 123, None, 1, 2, None, 4, None, 6]
    assert ak.to_list(three.mergemany([four, one, two])) == [100, 200, 300, None, None, 123, None, 1, 2, None, 4, None, 6, 7, None, None, 8, 9]
    assert ak.to_list(three.mergemany([two, one])) == [100, 200, 300, 7, None, None, 8, 9, 1, 2, None, 4, None, 6]
    assert ak.to_list(three.mergemany([two, one, four])) == [100, 200, 300, 7, None, None, 8, 9, 1, 2, None, 4, None, 6, None, None, 123, None]

def test_empty():
    one = ak.layout.EmptyArray()
    two = ak.layout.EmptyArray()
    three = ak.Array([1, 2, 3]).layout
    four = ak.Array([4, 5]).layout
    assert ak.to_list(one.mergemany([two])) == []
    assert ak.to_list(one.mergemany([two, one, two, one, two])) == []
    assert ak.to_list(one.mergemany([two, three])) == [1, 2, 3]
    assert ak.to_list(one.mergemany([two, three, four])) == [1, 2, 3, 4, 5]
    assert ak.to_list(one.mergemany([three])) == [1, 2, 3]
    assert ak.to_list(one.mergemany([three, four])) == [1, 2, 3, 4, 5]
    assert ak.to_list(one.mergemany([three, two])) == [1, 2, 3]
    assert ak.to_list(one.mergemany([three, two, four])) == [1, 2, 3, 4, 5]
    assert ak.to_list(one.mergemany([three, four, two])) == [1, 2, 3, 4, 5]

def test_union():
    one = ak.Array([1, 2, [], [3, 4]]).layout
    two = ak.Array([100, 200, 300]).layout
    three = ak.Array([{"x": 1}, {"x": 2}, 5, 6, 7]).layout

    assert ak.to_list(one.mergemany([two, three])) == [1, 2, [], [3, 4], 100, 200, 300, {"x": 1}, {"x": 2}, 5, 6, 7]
    assert ak.to_list(one.mergemany([three, two])) == [1, 2, [], [3, 4], {"x": 1}, {"x": 2}, 5, 6, 7, 100, 200, 300]
    assert ak.to_list(two.mergemany([one, three])) == [100, 200, 300, 1, 2, [], [3, 4], {"x": 1}, {"x": 2}, 5, 6, 7]
    assert ak.to_list(two.mergemany([three, one])) == [100, 200, 300, {"x": 1}, {"x": 2}, 5, 6, 7, 1, 2, [], [3, 4]]
    assert ak.to_list(three.mergemany([one, two])) == [{"x": 1}, {"x": 2}, 5, 6, 7, 1, 2, [], [3, 4], 100, 200, 300]
    assert ak.to_list(three.mergemany([two, one])) == [{"x": 1}, {"x": 2}, 5, 6, 7, 100, 200, 300, 1, 2, [], [3, 4]]

def test_union_option():
    one = ak.Array([1, 2, [], [3, 4]]).layout
    two = ak.Array([100, None, 300]).layout
    three = ak.Array([{"x": 1}, {"x": 2}, 5, 6, 7]).layout

    assert ak.to_list(one.mergemany([two, three])) == [1, 2, [], [3, 4], 100, None, 300, {"x": 1}, {"x": 2}, 5, 6, 7]
    assert ak.to_list(one.mergemany([three, two])) == [1, 2, [], [3, 4], {"x": 1}, {"x": 2}, 5, 6, 7, 100, None, 300]
    assert ak.to_list(two.mergemany([one, three])) == [100, None, 300, 1, 2, [], [3, 4], {"x": 1}, {"x": 2}, 5, 6, 7]
    assert ak.to_list(two.mergemany([three, one])) == [100, None, 300, {"x": 1}, {"x": 2}, 5, 6, 7, 1, 2, [], [3, 4]]
    assert ak.to_list(three.mergemany([one, two])) == [{"x": 1}, {"x": 2}, 5, 6, 7, 1, 2, [], [3, 4], 100, None, 300]
    assert ak.to_list(three.mergemany([two, one])) == [{"x": 1}, {"x": 2}, 5, 6, 7, 100, None, 300, 1, 2, [], [3, 4]]

    one = ak.Array([1, 2, [], [3, 4]]).layout
    two = ak.Array([100, None, 300]).layout
    three = ak.Array([{"x": 1}, {"x": 2}, 5, None, 7]).layout

    assert ak.to_list(one.mergemany([two, three])) == [1, 2, [], [3, 4], 100, None, 300, {"x": 1}, {"x": 2}, 5, None, 7]
    assert ak.to_list(one.mergemany([three, two])) == [1, 2, [], [3, 4], {"x": 1}, {"x": 2}, 5, None, 7, 100, None, 300]
    assert ak.to_list(two.mergemany([one, three])) == [100, None, 300, 1, 2, [], [3, 4], {"x": 1}, {"x": 2}, 5, None, 7]
    assert ak.to_list(two.mergemany([three, one])) == [100, None, 300, {"x": 1}, {"x": 2}, 5, None, 7, 1, 2, [], [3, 4]]
    assert ak.to_list(three.mergemany([one, two])) == [{"x": 1}, {"x": 2}, 5, None, 7, 1, 2, [], [3, 4], 100, None, 300]
    assert ak.to_list(three.mergemany([two, one])) == [{"x": 1}, {"x": 2}, 5, None, 7, 100, None, 300, 1, 2, [], [3, 4]]

    one = ak.Array([1, 2, [], [3, 4]]).layout
    two = ak.Array([100, 200, 300]).layout
    three = ak.Array([{"x": 1}, {"x": 2}, 5, None, 7]).layout

    assert ak.to_list(one.mergemany([two, three])) == [1, 2, [], [3, 4], 100, 200, 300, {"x": 1}, {"x": 2}, 5, None, 7]
    assert ak.to_list(one.mergemany([three, two])) == [1, 2, [], [3, 4], {"x": 1}, {"x": 2}, 5, None, 7, 100, 200, 300]
    assert ak.to_list(two.mergemany([one, three])) == [100, 200, 300, 1, 2, [], [3, 4], {"x": 1}, {"x": 2}, 5, None, 7]
    assert ak.to_list(two.mergemany([three, one])) == [100, 200, 300, {"x": 1}, {"x": 2}, 5, None, 7, 1, 2, [], [3, 4]]
    assert ak.to_list(three.mergemany([one, two])) == [{"x": 1}, {"x": 2}, 5, None, 7, 1, 2, [], [3, 4], 100, 200, 300]
    assert ak.to_list(three.mergemany([two, one])) == [{"x": 1}, {"x": 2}, 5, None, 7, 100, 200, 300, 1, 2, [], [3, 4]]

def test_strings():
    one = ak.Array(["uno", "dos", "tres"]).layout
    two = ak.Array(["un", "deux", "trois", "quatre"]).layout
    three = ak.Array(["onay", "ootay", "eethray"]).layout
    assert ak.to_list(one.mergemany([two, three])) == ["uno", "dos", "tres", "un", "deux", "trois", "quatre", "onay", "ootay", "eethray"]

def test_concatenate():
    one = ak.Array([1, 2, 3])
    two = ak.Array([4.4, 5.5])
    three = ak.Array([6, 7, 8])
    four = ak.Array([[9, 9, 9], [10, 10, 10]])
    assert ak.concatenate([one, two, three, four]).tolist() == [1, 2, 3, 4.4, 5.5, 6, 7, 8, [9, 9, 9], [10, 10, 10]]
    assert ak.concatenate([four, one, two, three]).tolist() == [[9, 9, 9], [10, 10, 10], 1, 2, 3, 4.4, 5.5, 6, 7, 8]
    assert ak.concatenate([one, two, four, three]).tolist() == [1, 2, 3, 4.4, 5.5, [9, 9, 9], [10, 10, 10], 6, 7, 8]

    five = ak.Array(["nine", "ten"])
    assert ak.concatenate([one, two, three, five]).tolist() == [1, 2, 3, 4.4, 5.5, 6, 7, 8, "nine", "ten"]
    assert ak.concatenate([five, one, two, three]).tolist() == ["nine", "ten", 1, 2, 3, 4.4, 5.5, 6, 7, 8]
    assert ak.concatenate([one, two, five, three]).tolist() == [1, 2, 3, 4.4, 5.5, "nine", "ten", 6, 7, 8]
