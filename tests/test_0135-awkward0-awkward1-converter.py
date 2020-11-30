# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward1 as ak  # noqa: F401

awkward0 = pytest.importorskip("awkward0")


def test_toawkward0():
    array = ak.from_iter([1.1, 2.2, 3.3, 4.4], highlevel=False)
    assert isinstance(ak.to_awkward0(array), np.ndarray)
    assert ak.to_awkward0(array).tolist() == [1.1, 2.2, 3.3, 4.4]

    array = ak.from_numpy(
        np.arange(2 * 3 * 5).reshape(2, 3, 5), highlevel=False
    ).toRegularArray()
    assert isinstance(ak.to_awkward0(array), awkward0.JaggedArray)
    assert ak.to_awkward0(array).tolist() == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]

    array = ak.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    assert isinstance(ak.to_awkward0(array), awkward0.JaggedArray)
    assert ak.to_awkward0(array).tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    array = ak.layout.ListArray64(
        ak.layout.Index64(np.array([4, 999, 1], dtype=np.int64)),
        ak.layout.Index64(np.array([7, 999, 3], dtype=np.int64)),
        ak.layout.NumpyArray(np.array([3.14, 4.4, 5.5, 123, 1.1, 2.2, 3.3, 321])),
    )
    assert isinstance(ak.to_awkward0(array), awkward0.JaggedArray)
    assert ak.to_awkward0(array).tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    array = ak.from_iter(
        [
            {"x": 0, "y": []},
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 3, 3]},
        ],
        highlevel=False,
    )
    assert isinstance(ak.to_awkward0(array[2]), dict)
    assert ak.to_awkward0(array[2])["x"] == 2.2
    assert isinstance(ak.to_awkward0(array[2])["y"], np.ndarray)
    assert ak.to_awkward0(array[2])["y"].tolist() == [2, 2]

    assert isinstance(ak.to_awkward0(array), awkward0.Table)
    assert ak.to_awkward0(array).tolist() == [
        {"x": 0, "y": []},
        {"x": 1.1, "y": [1]},
        {"x": 2.2, "y": [2, 2]},
        {"x": 3.3, "y": [3, 3, 3]},
    ]

    array = ak.from_iter(
        [(0, []), (1.1, [1]), (2.2, [2, 2]), (3.3, [3, 3, 3])], highlevel=False
    )
    assert isinstance(ak.to_awkward0(array), awkward0.Table)
    assert ak.to_awkward0(array).tolist() == [
        (0, []),
        (1.1, [1]),
        (2.2, [2, 2]),
        (3.3, [3, 3, 3]),
    ]
    assert isinstance(ak.to_awkward0(array[2]), tuple)
    assert ak.to_awkward0(array[2])[0] == 2.2
    assert ak.to_awkward0(array[2])[1].tolist() == [2, 2]

    array = ak.from_iter(
        [0.0, [], 1.1, [1], 2.2, [2, 2], 3.3, [3, 3, 3]], highlevel=False
    )
    assert isinstance(ak.to_awkward0(array), awkward0.UnionArray)
    assert ak.to_awkward0(array).tolist() == [
        0.0,
        [],
        1.1,
        [1],
        2.2,
        [2, 2],
        3.3,
        [3, 3, 3],
    ]

    array = ak.from_iter([1.1, 2.2, None, None, 3.3, None, 4.4], highlevel=False)
    assert isinstance(ak.to_awkward0(array), awkward0.IndexedMaskedArray)
    assert ak.to_awkward0(array).tolist() == [1.1, 2.2, None, None, 3.3, None, 4.4]

    content = ak.layout.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    index = ak.layout.Index64(np.array([3, 2, 2, 5, 0], dtype=np.int64))
    array = ak.layout.IndexedArray64(index, content)
    assert isinstance(ak.to_awkward0(array), awkward0.IndexedArray)
    assert ak.to_awkward0(array).tolist() == [3.3, 2.2, 2.2, 5.5, 0.0]


def test_fromawkward0():
    array = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    assert isinstance(ak.from_awkward0(array), ak.highlevel.Array)
    assert isinstance(ak.from_awkward0(array, highlevel=False), ak.layout.NumpyArray)
    assert ak.to_list(array) == [1.1, 2.2, 3.3, 4.4, 5.5]

    array = (123, np.array([1.1, 2.2, 3.3]))
    assert isinstance(ak.from_awkward0(array), ak.highlevel.Record)
    assert isinstance(ak.from_awkward0(array).layout, ak.layout.Record)
    assert ak.to_list(ak.from_awkward0(array)) == (123, [1.1, 2.2, 3.3])

    array = {"x": 123, "y": np.array([1.1, 2.2, 3.3])}
    assert isinstance(ak.from_awkward0(array), ak.highlevel.Record)
    assert isinstance(ak.from_awkward0(array).layout, ak.layout.Record)
    assert ak.to_list(ak.from_awkward0(array)) == {"x": 123, "y": [1.1, 2.2, 3.3]}

    array = awkward0.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    assert isinstance(
        ak.from_awkward0(array, highlevel=False),
        (
            ak.layout.ListOffsetArray32,
            ak.layout.ListOffsetArrayU32,
            ak.layout.ListOffsetArray64,
        ),
    )
    assert ak.to_list(ak.from_awkward0(array)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    array = awkward0.fromiter(
        [{"x": 0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}]
    )
    assert isinstance(ak.from_awkward0(array, highlevel=False), ak.layout.RecordArray)
    assert not ak.from_awkward0(array, highlevel=False).istuple
    assert ak.from_awkward0(array).layout.keys() == ["x", "y"]
    assert ak.to_list(ak.from_awkward0(array)) == [
        {"x": 0, "y": []},
        {"x": 1.1, "y": [1]},
        {"x": 2.2, "y": [2, 2]},
    ]

    array = awkward0.Table([0.0, 1.1, 2.2], awkward0.fromiter([[], [1], [2, 2]]))
    assert isinstance(ak.from_awkward0(array, highlevel=False), ak.layout.RecordArray)
    assert ak.from_awkward0(array, highlevel=False).istuple
    assert ak.from_awkward0(array).layout.keys() == ["0", "1"]
    assert ak.to_list(ak.from_awkward0(array)) == [(0.0, []), (1.1, [1]), (2.2, [2, 2])]

    array = awkward0.fromiter([0.0, [], 1.1, [1], 2.2, [2, 2], 3.3, [3, 3, 3]])
    assert isinstance(
        ak.from_awkward0(array, highlevel=False),
        (ak.layout.UnionArray8_32, ak.layout.UnionArray8_U32, ak.layout.UnionArray8_64),
    )
    assert ak.to_list(ak.from_awkward0(array)) == [
        0.0,
        [],
        1.1,
        [1],
        2.2,
        [2, 2],
        3.3,
        [3, 3, 3],
    ]

    array = awkward0.fromiter([1.1, 2.2, None, None, 3.3, None, 4.4])
    assert isinstance(
        ak.from_awkward0(array, highlevel=False), ak.layout.ByteMaskedArray
    )
    assert ak.to_list(ak.from_awkward0(array)) == [1.1, 2.2, None, None, 3.3, None, 4.4]

    array = awkward0.fromiter(["hello", "you", "guys"])
    assert isinstance(
        ak.from_awkward0(array, highlevel=False),
        (
            ak.layout.ListArray32,
            ak.layout.ListArrayU32,
            ak.layout.ListArray64,
            ak.layout.ListOffsetArray32,
            ak.layout.ListOffsetArrayU32,
            ak.layout.ListOffsetArray64,
        ),
    )
    assert ak.from_awkward0(array, highlevel=False).parameters["__array__"] in (
        "string",
        "bytestring",
    )
    assert ak.from_awkward0(array, highlevel=False).content.parameters["__array__"] in (
        "char",
        "byte",
    )
    assert ak.to_list(ak.from_awkward0(array)) == ["hello", "you", "guys"]

    class Point(object):
        def __init__(self, x, y):
            self.x, self.y = x, y

        def __repr__(self):
            return "Point({0}, {1})".format(self.x, self.y)

    array = awkward0.fromiter([Point(1.1, 10), Point(2.2, 20), Point(3.3, 30)])
    assert ak.to_list(ak.from_awkward0(array)) == [
        {"x": 1.1, "y": 10},
        {"x": 2.2, "y": 20},
        {"x": 3.3, "y": 30},
    ]
    assert "__record__" in ak.from_awkward0(array).layout.parameters

    array = awkward0.ChunkedArray(
        [
            awkward0.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]]),
            awkward0.fromiter([[6.6]]),
            awkward0.fromiter([[7.7, 8.8], [9.9, 10.0, 11.1, 12.2]]),
        ]
    )
    assert ak.to_list(ak.from_awkward0(array)) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8],
        [9.9, 10.0, 11.1, 12.2],
    ]

    def generate1():
        return awkward0.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

    def generate2():
        return awkward0.fromiter([[6.6]])

    def generate3():
        return awkward0.fromiter([[7.7, 8.8], [9.9, 10.0, 11.1, 12.2]])

    array = awkward0.ChunkedArray(
        [
            awkward0.VirtualArray(generate1),
            awkward0.VirtualArray(generate2),
            awkward0.VirtualArray(generate3),
        ]
    )
    assert ak.to_list(ak.from_awkward0(array)) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8],
        [9.9, 10.0, 11.1, 12.2],
    ]
