from __future__ import annotations

import numpy as np

import awkward as ak

to_list = ak.operations.to_list


def test_0724():
    a = ak.contents.NumpyArray(np.empty(0))
    idx = ak.index.Index64([])
    a = ak.contents.IndexedOptionArray(idx, a)
    idx = ak.index.Index64([0])
    a = ak.contents.ListOffsetArray(idx, a)
    idx = ak.index.Index64([175990832])
    a = ak.contents.ListOffsetArray(idx, a)

    assert ak.operations.flatten(a, axis=2).to_list() == []
    assert str(ak.operations.flatten(a, axis=2).type) == "0 * var * ?float64"


def test_ByteMaskedArray_flatten():
    content = ak.operations.from_iter(
        [
            [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 7.7, 8.8, 9.9]],
            [[], [10.0, 11.1, 12.2]],
        ],
        highlevel=False,
    )
    mask = ak.index.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.contents.ByteMaskedArray(mask, content, valid_when=False)

    assert to_list(array) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array, axis=1)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [],
        [10.0, 11.1, 12.2],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array, axis=-2)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [],
        [10.0, 11.1, 12.2],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array, axis=2)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        None,
        None,
        [10.0, 11.1, 12.2],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array, axis=-1)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        None,
        None,
        [10.0, 11.1, 12.2],
    ]


def test_flatten_ListOffsetArray():
    array = ak.highlevel.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout

    assert ak.operations.to_list(ak.operations.flatten(array)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[1:])) == [4.4, 5.5]

    array = ak.highlevel.Array(
        [[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]], [[], [6.6, 7.7, 8.8, 9.9]]]
    ).layout

    assert ak.operations.to_list(ak.operations.flatten(array)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[1:])) == [
        [5.5],
        [],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[:, 1:])) == [
        [],
        [3.3, 4.4],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array, axis=2)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[1:], axis=2)) == [
        [],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[:, 1:], axis=2)) == [
        [3.3, 4.4],
        [],
        [],
        [6.6, 7.7, 8.8, 9.9],
    ]

    array = ak.highlevel.Array(
        np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7).tolist()
    ).layout

    assert (
        ak.operations.to_list(ak.operations.flatten(array, axis=1))
        == np.arange(2 * 3 * 5 * 7).reshape(2 * 3, 5, 7).tolist()
    )
    assert (
        ak.operations.to_list(ak.operations.flatten(array, axis=2))
        == np.arange(2 * 3 * 5 * 7).reshape(2, 3 * 5, 7).tolist()
    )
    assert (
        ak.operations.to_list(ak.operations.flatten(array, axis=3))
        == np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5 * 7).tolist()
    )

    array = ak.highlevel.Array(
        ak.operations.from_iter(
            np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7).tolist(), highlevel=False
        )
    ).layout

    assert (
        ak.operations.to_list(ak.operations.flatten(array, axis=1))
        == np.arange(2 * 3 * 5 * 7).reshape(2 * 3, 5, 7).tolist()
    )
    assert (
        ak.operations.to_list(ak.operations.flatten(array, axis=2))
        == np.arange(2 * 3 * 5 * 7).reshape(2, 3 * 5, 7).tolist()
    )
    assert (
        ak.operations.to_list(ak.operations.flatten(array, axis=3))
        == np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5 * 7).tolist()
    )

    array = ak.highlevel.Array(np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)).layout

    assert (
        ak.operations.to_list(ak.operations.flatten(array, axis=1))
        == np.arange(2 * 3 * 5 * 7).reshape(2 * 3, 5, 7).tolist()
    )
    assert (
        ak.operations.to_list(ak.operations.flatten(array, axis=2))
        == np.arange(2 * 3 * 5 * 7).reshape(2, 3 * 5, 7).tolist()
    )
    assert (
        ak.operations.to_list(ak.operations.flatten(array, axis=3))
        == np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5 * 7).tolist()
    )


def test_flatten_IndexedArray():
    array = ak.highlevel.Array(
        [[1.1, 2.2, None, 3.3], None, [], None, [4.4, 5.5], None]
    ).layout

    assert ak.operations.to_list(ak.operations.flatten(array)) == [
        1.1,
        2.2,
        None,
        3.3,
        4.4,
        5.5,
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[1:])) == [4.4, 5.5]

    array = ak.highlevel.Array(
        [
            [[0.0, 1.1, 2.2], None, None, [3.3, 4.4]],
            [],
            [[5.5]],
            [[], [6.6, 7.7, 8.8, 9.9]],
        ]
    ).layout

    assert ak.operations.to_list(ak.operations.flatten(array, axis=2)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[1:], axis=2)) == [
        [],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[:, 1:], axis=2)) == [
        [3.3, 4.4],
        [],
        [],
        [6.6, 7.7, 8.8, 9.9],
    ]

    array = ak.highlevel.Array(
        [
            [[0.0, 1.1, 2.2], [3.3, 4.4]],
            [],
            [[5.5]],
            None,
            None,
            [[], [6.6, 7.7, 8.8, 9.9]],
        ]
    ).layout

    assert ak.operations.to_list(ak.operations.flatten(array, axis=2)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        [5.5],
        None,
        None,
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[1:], axis=2)) == [
        [],
        [5.5],
        None,
        None,
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[:, 1:], axis=2)) == [
        [3.3, 4.4],
        [],
        [],
        None,
        None,
        [6.6, 7.7, 8.8, 9.9],
    ]

    array = ak.highlevel.Array(
        [
            [[0.0, 1.1, None, 2.2], None, [], None, [3.3, 4.4]],
            None,
            [],
            [[5.5]],
            None,
            [[], [6.6, None, 7.7, 8.8, 9.9], None],
        ]
    ).layout

    assert ak.operations.to_list(ak.operations.flatten(array)) == [
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
    assert ak.operations.to_list(ak.operations.flatten(array, axis=2)) == [
        [0.0, 1.1, None, 2.2, 3.3, 4.4],
        None,
        [],
        [5.5],
        None,
        [6.6, None, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[1:], axis=2)) == [
        None,
        [],
        [5.5],
        None,
        [6.6, None, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[:, 1:], axis=2)) == [
        [3.3, 4.4],
        None,
        [],
        [],
        None,
        [6.6, None, 7.7, 8.8, 9.9],
    ]

    content = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    index = ak.index.Index64(np.array([2, 1, 0, 3, 3, 4], dtype=np.int64))
    array = ak.contents.IndexedArray(index, content)

    assert to_list(array) == [
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
        [5.5],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array)) == [
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

    content = ak.operations.from_iter(
        [[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]], [[], [6.6, 7.7, 8.8, 9.9]]],
        highlevel=False,
    )
    index = ak.index.Index64(np.array([2, 2, 1, 0, 3], dtype=np.int64))
    array = ak.contents.IndexedArray(index, content)

    assert to_list(array) == [
        [[5.5]],
        [[5.5]],
        [],
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [[], [6.6, 7.7, 8.8, 9.9]],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array, axis=2)) == [
        [5.5],
        [5.5],
        [],
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [6.6, 7.7, 8.8, 9.9],
    ]


def test_flatten_RecordArray():
    array = ak.highlevel.Array(
        [
            {"x": [], "y": [[3, 3, 3]]},
            {"x": [[1]], "y": [[2, 2]]},
            {"x": [[2], [2]], "y": [[1]]},
            {"x": [[3], [3], [3]], "y": [[]]},
        ]
    ).layout

    assert ak.operations.to_list(ak.operations.flatten(array, axis=2)) == [
        {"x": [], "y": [3, 3, 3]},
        {"x": [1], "y": [2, 2]},
        {"x": [2, 2], "y": [1]},
        {"x": [3, 3, 3], "y": []},
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[1:], axis=2)) == [
        {"x": [1], "y": [2, 2]},
        {"x": [2, 2], "y": [1]},
        {"x": [3, 3, 3], "y": []},
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[:, 1:], axis=2)) == [
        {"x": [], "y": []},
        {"x": [], "y": []},
        {"x": [2], "y": []},
        {"x": [3, 3], "y": []},
    ]


def test_flatten_UnionArray():
    content1 = ak.operations.from_iter(
        [[1.1], [2.2, 2.2], [3.3, 3.3, 3.3]], highlevel=False
    )
    content2 = ak.operations.from_iter(
        [[[3, 3, 3], [3, 3, 3], [3, 3, 3]], [[2, 2], [2, 2]], [[1]]], highlevel=False
    )
    content3 = ak.operations.from_iter(
        [
            [["3", "3", "3"], ["3", "3", "3"], ["3", "3", "3"]],
            [["2", "2"], ["2", "2"]],
            [["1"]],
        ],
        highlevel=False,
    )
    tags = ak.index.Index8(np.array([0, 1, 0, 1, 0, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0, 1, 1, 2, 2], dtype=np.int64))
    array = ak.contents.UnionArray(tags, index, [content1, content2])
    array = ak.to_backend(array, "cuda")

    assert to_list(array) == [
        [1.1],
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [2.2, 2.2],
        [[2, 2], [2, 2]],
        [3.3, 3.3, 3.3],
        [[1]],
    ]
    assert to_list(array[1:]) == [
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [2.2, 2.2],
        [[2, 2], [2, 2]],
        [3.3, 3.3, 3.3],
        [[1]],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array)) == [
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
    assert ak.operations.to_list(ak.operations.flatten(array[1:])) == [
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

    array = ak.contents.UnionArray(tags, index, [content2, content3])

    assert to_list(array) == [
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [["3", "3", "3"], ["3", "3", "3"], ["3", "3", "3"]],
        [[2, 2], [2, 2]],
        [["2", "2"], ["2", "2"]],
        [[1]],
        [["1"]],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array, axis=2)) == [
        [3, 3, 3, 3, 3, 3, 3, 3, 3],
        ["3", "3", "3", "3", "3", "3", "3", "3", "3"],
        [2, 2, 2, 2],
        ["2", "2", "2", "2"],
        [1],
        ["1"],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[1:], axis=2)) == [
        ["3", "3", "3", "3", "3", "3", "3", "3", "3"],
        [2, 2, 2, 2],
        ["2", "2", "2", "2"],
        [1],
        ["1"],
    ]
    assert ak.operations.to_list(ak.operations.flatten(array[:, 1:], axis=2)) == [
        [3, 3, 3, 3, 3, 3],
        ["3", "3", "3", "3", "3", "3"],
        [2, 2],
        ["2", "2"],
        [],
        [],
    ]
