# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak


def test_emptyarray():
    array = ak.layout.EmptyArray()
    assert ak.to_list(array.num(0)) == 0
    assert ak.to_list(array.num(1)) == []
    assert ak.to_list(array.num(2)) == []


def test_numpyarray():
    array = ak.layout.NumpyArray(np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7))
    assert array.num(0) == 2
    assert ak.to_list(array.num(1)) == [3, 3]
    assert ak.to_list(array.num(2)) == [[5, 5, 5], [5, 5, 5]]
    assert ak.to_list(array.num(3)) == [
        [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]],
        [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]],
    ]
    with pytest.raises(ValueError) as err:
        array.num(4)
    assert str(err.value).startswith("'axis' out of range for 'num'")


def test_regulararray():
    array = ak.layout.NumpyArray(
        np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    ).toRegularArray()
    assert array.num(0) == 2
    assert ak.to_list(array.num(1)) == [3, 3]
    assert ak.to_list(array.num(2)) == [[5, 5, 5], [5, 5, 5]]
    assert ak.to_list(array.num(3)) == [
        [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]],
        [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]],
    ]
    with pytest.raises(ValueError) as err:
        array.num(4)
    assert str(err.value).startswith("'axis' out of range for 'num'")


def test_listarray():
    content = ak.layout.NumpyArray(np.arange(2 * 3 * 5).reshape(5, 3, 2))
    starts = ak.layout.Index64(np.array([0, 3, 3], dtype=np.int64))
    stops = ak.layout.Index64(np.array([3, 3, 5], dtype=np.int64))
    array = ak.layout.ListArray64(starts, stops, content)
    assert ak.to_list(array) == [
        [
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
            [[12, 13], [14, 15], [16, 17]],
        ],
        [],
        [[[18, 19], [20, 21], [22, 23]], [[24, 25], [26, 27], [28, 29]]],
    ]

    assert ak.to_list(array.num(0)) == 3
    assert ak.to_list(array.num(1)) == [3, 0, 2]
    assert ak.to_list(array.num(2)) == [[3, 3, 3], [], [3, 3]]
    assert ak.to_list(array.num(3)) == [
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
        [],
        [[2, 2, 2], [2, 2, 2]],
    ]
    with pytest.raises(ValueError) as err:
        array.num(4)
    assert str(err.value).startswith("'axis' out of range for 'num'")


def test_listoffsetarray():
    content = ak.layout.NumpyArray(np.arange(2 * 3 * 5).reshape(5, 3, 2))
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    array = ak.layout.ListOffsetArray64(offsets, content)
    assert ak.to_list(array) == [
        [
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
            [[12, 13], [14, 15], [16, 17]],
        ],
        [],
        [[[18, 19], [20, 21], [22, 23]], [[24, 25], [26, 27], [28, 29]]],
    ]

    assert ak.to_list(array.num(0)) == 3
    assert ak.to_list(array.num(1)) == [3, 0, 2]
    assert ak.to_list(array.num(2)) == [[3, 3, 3], [], [3, 3]]
    assert ak.to_list(array.num(3)) == [
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
        [],
        [[2, 2, 2], [2, 2, 2]],
    ]
    with pytest.raises(ValueError) as err:
        array.num(4)
    assert str(err.value).startswith("'axis' out of range for 'num'")


def test_indexedarray():
    content = ak.layout.NumpyArray(np.arange(2 * 3 * 5).reshape(5, 3, 2))
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    listarray = ak.layout.ListOffsetArray64(offsets, content)
    index = ak.layout.Index64(np.array([2, 2, 1, 0], dtype=np.int64))
    array = ak.layout.IndexedArray64(index, listarray)
    assert ak.to_list(array) == [
        [[[18, 19], [20, 21], [22, 23]], [[24, 25], [26, 27], [28, 29]]],
        [[[18, 19], [20, 21], [22, 23]], [[24, 25], [26, 27], [28, 29]]],
        [],
        [
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
            [[12, 13], [14, 15], [16, 17]],
        ],
    ]

    assert ak.to_list(array.num(0)) == 4
    assert ak.to_list(array.num(1)) == [2, 2, 0, 3]
    assert ak.to_list(array.num(2)) == [[3, 3], [3, 3], [], [3, 3, 3]]
    assert ak.to_list(array.num(3)) == [
        [[2, 2, 2], [2, 2, 2]],
        [[2, 2, 2], [2, 2, 2]],
        [],
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
    ]

    with pytest.raises(ValueError) as err:
        array.num(4)
    assert str(err.value).startswith("'axis' out of range for 'num'")


def test_indexedoptionarray():
    content = ak.layout.NumpyArray(np.arange(2 * 3 * 5).reshape(5, 3, 2))
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    listarray = ak.layout.ListOffsetArray64(offsets, content)
    index = ak.layout.Index64(np.array([2, -1, 2, 1, -1, 0], dtype=np.int64))
    array = ak.layout.IndexedOptionArray64(index, listarray)
    assert ak.to_list(array) == [
        [[[18, 19], [20, 21], [22, 23]], [[24, 25], [26, 27], [28, 29]]],
        None,
        [[[18, 19], [20, 21], [22, 23]], [[24, 25], [26, 27], [28, 29]]],
        [],
        None,
        [
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
            [[12, 13], [14, 15], [16, 17]],
        ],
    ]

    assert ak.to_list(array.num(0)) == 6
    assert ak.to_list(array.num(1)) == [2, None, 2, 0, None, 3]
    assert ak.to_list(array.num(2)) == [[3, 3], None, [3, 3], [], None, [3, 3, 3]]
    assert ak.to_list(array.num(3)) == [
        [[2, 2, 2], [2, 2, 2]],
        None,
        [[2, 2, 2], [2, 2, 2]],
        [],
        None,
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
    ]

    with pytest.raises(ValueError) as err:
        array.num(4)
    assert str(err.value).startswith("'axis' out of range for 'num'")


def test_recordarray():
    array = ak.from_iter(
        [
            {"x": 0.0, "y": []},
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 3, 3]},
        ],
        highlevel=False,
    )
    assert ak.to_list(array.num(0)) == {"x": 4, "y": 4}

    array = ak.from_iter(
        [
            {"x": [3.3, 3.3, 3.3], "y": []},
            {"x": [2.2, 2.2], "y": [1]},
            {"x": [1.1], "y": [2, 2]},
            {"x": [], "y": [3, 3, 3]},
        ],
        highlevel=False,
    )
    assert ak.to_list(array.num(0)) == {"x": 4, "y": 4}
    assert ak.to_list(array.num(1)) == [
        {"x": 3, "y": 0},
        {"x": 2, "y": 1},
        {"x": 1, "y": 2},
        {"x": 0, "y": 3},
    ]
    assert ak.to_list(array.num(1)[2]) == {"x": 1, "y": 2}

    array = ak.from_iter(
        [
            {"x": [[3.3, 3.3, 3.3]], "y": []},
            {"x": [[2.2, 2.2]], "y": [1]},
            {"x": [[1.1]], "y": [2, 2]},
            {"x": [[]], "y": [3, 3, 3]},
        ],
        highlevel=False,
    )
    assert ak.to_list(array.num(0)) == {"x": 4, "y": 4}
    assert ak.to_list(array.num(1)) == [
        {"x": 1, "y": 0},
        {"x": 1, "y": 1},
        {"x": 1, "y": 2},
        {"x": 1, "y": 3},
    ]
    assert ak.to_list(array.num(1)[2]) == {"x": 1, "y": 2}


def test_unionarray():
    content1 = ak.from_iter([[], [1], [2, 2], [3, 3, 3]], highlevel=False)
    content2 = ak.from_iter([[3.3, 3.3, 3.3], [2.2, 2.2], [1.1], []], highlevel=False)
    tags = ak.layout.Index8(np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8))
    index = ak.layout.Index64(np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64))
    array = ak.layout.UnionArray8_64(tags, index, [content1, content2])
    assert ak.to_list(array) == [
        [],
        [3.3, 3.3, 3.3],
        [1],
        [2.2, 2.2],
        [2, 2],
        [1.1],
        [3, 3, 3],
        [],
    ]

    assert array.num(0) == 8
    assert isinstance(array.num(1), ak.layout.NumpyArray)
    assert ak.to_list(array.num(1)) == [0, 3, 1, 2, 2, 1, 3, 0]


def test_highlevel():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    assert ak.to_list(ak.num(array)) == [3, 0, 2]


def test_flatten_ListOffsetArray():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    assert ak.to_list(ak.flatten(array)) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert ak.to_list(ak.flatten(array[1:])) == [4.4, 5.5]

    array = ak.Array(
        [[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]], [[], [6.6, 7.7, 8.8, 9.9]]]
    )
    assert ak.to_list(ak.flatten(array)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(ak.flatten(array[1:])) == [[5.5], [], [6.6, 7.7, 8.8, 9.9]]
    assert ak.to_list(ak.flatten(array[:, 1:])) == [
        [],
        [3.3, 4.4],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(ak.flatten(array, axis=2)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(ak.flatten(array[1:], axis=2)) == [
        [],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(ak.flatten(array[:, 1:], axis=2)) == [
        [3.3, 4.4],
        [],
        [],
        [6.6, 7.7, 8.8, 9.9],
    ]

    array = ak.Array(np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7).tolist())
    assert (
        ak.to_list(ak.flatten(array, axis=1))
        == np.arange(2 * 3 * 5 * 7).reshape(2 * 3, 5, 7).tolist()
    )
    assert (
        ak.to_list(ak.flatten(array, axis=2))
        == np.arange(2 * 3 * 5 * 7).reshape(2, 3 * 5, 7).tolist()
    )
    assert (
        ak.to_list(ak.flatten(array, axis=3))
        == np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5 * 7).tolist()
    )

    def toListArray(x):
        if isinstance(x, ak.layout.ListOffsetArray64):
            starts = ak.layout.Index64(np.asarray(x.offsets)[:-1])
            stops = ak.layout.Index64(np.asarray(x.offsets)[1:])
            return ak.layout.ListArray64(starts, stops, toListArray(x.content))
        elif isinstance(x, ak.layout.ListOffsetArray32):
            starts = ak.layout.Index64(np.asarray(x.offsets)[:-1])
            stops = ak.layout.Index64(np.asarray(x.offsets)[1:])
            return ak.layout.ListArray32(starts, stops, toListArray(x.content))
        elif isinstance(x, ak.layout.ListOffsetArrayU32):
            starts = ak.layout.Index64(np.asarray(x.offsets)[:-1])
            stops = ak.layout.Index64(np.asarray(x.offsets)[1:])
            return ak.layout.ListArrayU32(starts, stops, toListArray(x.content))
        else:
            return x

    array = ak.Array(
        toListArray(
            ak.from_iter(
                np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7).tolist(), highlevel=False
            )
        )
    )
    assert (
        ak.to_list(ak.flatten(array, axis=1))
        == np.arange(2 * 3 * 5 * 7).reshape(2 * 3, 5, 7).tolist()
    )
    assert (
        ak.to_list(ak.flatten(array, axis=2))
        == np.arange(2 * 3 * 5 * 7).reshape(2, 3 * 5, 7).tolist()
    )
    assert (
        ak.to_list(ak.flatten(array, axis=3))
        == np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5 * 7).tolist()
    )

    array = ak.Array(np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7))
    assert (
        ak.to_list(ak.flatten(array, axis=1))
        == np.arange(2 * 3 * 5 * 7).reshape(2 * 3, 5, 7).tolist()
    )
    assert (
        ak.to_list(ak.flatten(array, axis=2))
        == np.arange(2 * 3 * 5 * 7).reshape(2, 3 * 5, 7).tolist()
    )
    assert (
        ak.to_list(ak.flatten(array, axis=3))
        == np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5 * 7).tolist()
    )


def test_flatten_IndexedArray():
    array = ak.Array([[1.1, 2.2, None, 3.3], None, [], None, [4.4, 5.5], None])
    assert ak.to_list(ak.flatten(array)) == [1.1, 2.2, None, 3.3, 4.4, 5.5]
    assert ak.to_list(ak.flatten(array[1:])) == [4.4, 5.5]

    array = ak.Array(
        [
            [[0.0, 1.1, 2.2], None, None, [3.3, 4.4]],
            [],
            [[5.5]],
            [[], [6.6, 7.7, 8.8, 9.9]],
        ]
    )
    assert ak.to_list(ak.flatten(array, axis=2)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(ak.flatten(array[1:], axis=2)) == [
        [],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(ak.flatten(array[:, 1:], axis=2)) == [
        [3.3, 4.4],
        [],
        [],
        [6.6, 7.7, 8.8, 9.9],
    ]

    array = ak.Array(
        [
            [[0.0, 1.1, 2.2], [3.3, 4.4]],
            [],
            [[5.5]],
            None,
            None,
            [[], [6.6, 7.7, 8.8, 9.9]],
        ]
    )
    assert ak.to_list(ak.flatten(array, axis=2)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        [5.5],
        None,
        None,
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(ak.flatten(array[1:], axis=2)) == [
        [],
        [5.5],
        None,
        None,
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(ak.flatten(array[:, 1:], axis=2)) == [
        [3.3, 4.4],
        [],
        [],
        None,
        None,
        [6.6, 7.7, 8.8, 9.9],
    ]

    array = ak.Array(
        [
            [[0.0, 1.1, None, 2.2], None, [], None, [3.3, 4.4]],
            None,
            [],
            [[5.5]],
            None,
            [[], [6.6, None, 7.7, 8.8, 9.9], None],
        ]
    )
    assert ak.to_list(ak.flatten(array)) == [
        [0.0, 1.1, None, 2.2],
        None,
        [],
        None,
        [3.3, 4.4],
        [5.5],
        [],
        [6.6, None, 7.7, 8.8, 9.9],
        None,
    ]
    assert ak.to_list(ak.flatten(array, axis=2)) == [
        [0.0, 1.1, None, 2.2, 3.3, 4.4],
        None,
        [],
        [5.5],
        None,
        [6.6, None, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(ak.flatten(array[1:], axis=2)) == [
        None,
        [],
        [5.5],
        None,
        [6.6, None, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(ak.flatten(array[:, 1:], axis=2)) == [
        [3.3, 4.4],
        None,
        [],
        [],
        None,
        [6.6, None, 7.7, 8.8, 9.9],
    ]

    content = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    index = ak.layout.Index64(np.array([2, 1, 0, 3, 3, 4], dtype=np.int64))
    array = ak.Array(ak.layout.IndexedArray64(index, content))
    assert ak.to_list(array) == [
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
        [5.5],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(ak.flatten(array)) == [
        3.3,
        4.4,
        0.0,
        1.1,
        2.2,
        5.5,
        5.5,
        6.6,
        7.7,
        8.8,
        9.9,
    ]

    content = ak.from_iter(
        [[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]], [[], [6.6, 7.7, 8.8, 9.9]]],
        highlevel=False,
    )
    index = ak.layout.Index64(np.array([2, 2, 1, 0, 3], dtype=np.int64))
    array = ak.Array(ak.layout.IndexedArray64(index, content))
    assert ak.to_list(array) == [
        [[5.5]],
        [[5.5]],
        [],
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [[], [6.6, 7.7, 8.8, 9.9]],
    ]
    assert ak.to_list(ak.flatten(array, axis=2)) == [
        [5.5],
        [5.5],
        [],
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [6.6, 7.7, 8.8, 9.9],
    ]


def test_flatten_RecordArray():
    array = ak.Array(
        [
            {"x": [], "y": [[3, 3, 3]]},
            {"x": [[1]], "y": [[2, 2]]},
            {"x": [[2], [2]], "y": [[1]]},
            {"x": [[3], [3], [3]], "y": [[]]},
        ]
    )
    assert ak.to_list(ak.flatten(array, axis=2)) == [
        {"x": [], "y": [3, 3, 3]},
        {"x": [1], "y": [2, 2]},
        {"x": [2, 2], "y": [1]},
        {"x": [3, 3, 3], "y": []},
    ]
    assert ak.to_list(ak.flatten(array[1:], axis=2)) == [
        {"x": [1], "y": [2, 2]},
        {"x": [2, 2], "y": [1]},
        {"x": [3, 3, 3], "y": []},
    ]
    assert ak.to_list(ak.flatten(array[:, 1:], axis=2)) == [
        {"x": [], "y": []},
        {"x": [], "y": []},
        {"x": [2], "y": []},
        {"x": [3, 3], "y": []},
    ]


def test_flatten_UnionArray():
    content1 = ak.from_iter([[1.1], [2.2, 2.2], [3.3, 3.3, 3.3]], highlevel=False)
    content2 = ak.from_iter(
        [[[3, 3, 3], [3, 3, 3], [3, 3, 3]], [[2, 2], [2, 2]], [[1]]], highlevel=False
    )
    tags = ak.layout.Index8(np.array([0, 1, 0, 1, 0, 1], dtype=np.int8))
    index = ak.layout.Index64(np.array([0, 0, 1, 1, 2, 2], dtype=np.int64))
    array = ak.Array(ak.layout.UnionArray8_64(tags, index, [content1, content2]))
    assert ak.to_list(array) == [
        [1.1],
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [2.2, 2.2],
        [[2, 2], [2, 2]],
        [3.3, 3.3, 3.3],
        [[1]],
    ]
    assert ak.to_list(array[1:]) == [
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [2.2, 2.2],
        [[2, 2], [2, 2]],
        [3.3, 3.3, 3.3],
        [[1]],
    ]
    assert ak.to_list(ak.flatten(array)) == [
        1.1,
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        2.2,
        2.2,
        [2, 2],
        [2, 2],
        3.3,
        3.3,
        3.3,
        [1],
    ]
    assert ak.to_list(ak.flatten(array[1:])) == [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        2.2,
        2.2,
        [2, 2],
        [2, 2],
        3.3,
        3.3,
        3.3,
        [1],
    ]

    array = ak.Array(ak.layout.UnionArray8_64(tags, index, [content2, content2]))
    assert ak.to_list(array) == [
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [[2, 2], [2, 2]],
        [[2, 2], [2, 2]],
        [[1]],
        [[1]],
    ]
    assert ak.to_list(ak.flatten(array, axis=2)) == [
        [3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [1],
        [1],
    ]
    assert ak.to_list(ak.flatten(array[1:], axis=2)) == [
        [3, 3, 3, 3, 3, 3, 3, 3, 3],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [1],
        [1],
    ]
    assert ak.to_list(ak.flatten(array[:, 1:], axis=2)) == [
        [3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3],
        [2, 2],
        [2, 2],
        [],
        [],
    ]
