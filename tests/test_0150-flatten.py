# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list

"""includes test_0117, test_0110, test_0042, test_0127,
 test_0198, test_0446, test_0585, test_0590, test_0612,
 test_0724, test_0866, test_0973
"""


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


def test_flatten_0042():
    array = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True
    ).layout
    assert ak.operations.to_list(ak.operations.flatten(array, axis=1)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]


def test_flatten2():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10], dtype=np.int64))
    array = ak.contents.ListOffsetArray(offsets, content)

    assert to_list(array) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert to_list(ak._do.flatten(array, axis=1)) == [
        0.0,
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        6.6,
        7.7,
        8.8,
        9.9,
    ]
    assert to_list(ak._do.flatten(array, axis=-1)) == [
        0.0,
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        6.6,
        7.7,
        8.8,
        9.9,
    ]
    with pytest.raises(ValueError) as err:
        assert to_list(ak._do.flatten(array, axis=-2))
        assert str(err.value).startswith("axis=0 not allowed for flatten")

    array2 = array[2:-1]
    assert to_list(ak._do.flatten(array2, axis=1)) == [3.3, 4.4, 5.5]
    assert to_list(ak._do.flatten(array2, axis=-1)) == [3.3, 4.4, 5.5]


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


def test_flatten_0198():
    array = ak.highlevel.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5]).layout

    assert ak.operations.to_list(ak.operations.flatten(array, axis=0)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]

    content0 = ak.operations.from_iter(
        [1.1, 2.2, None, 3.3, None, None, 4.4, 5.5], highlevel=False
    )
    content1 = ak.operations.from_iter(
        ["one", None, "two", None, "three"], highlevel=False
    )
    array = ak.contents.UnionArray(
        ak.index.Index8(
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0], dtype=np.int8)
        ),
        ak.index.Index64(
            np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 4, 7], dtype=np.int64)
        ),
        [content0, content1],
    )

    assert to_list(array) == [
        1.1,
        "one",
        2.2,
        None,
        None,
        "two",
        3.3,
        None,
        None,
        None,
        4.4,
        "three",
        5.5,
    ]
    assert ak.operations.to_list(ak.operations.flatten(array, axis=0)) == [
        1.1,
        "one",
        2.2,
        "two",
        3.3,
        4.4,
        "three",
        5.5,
    ]


def test_fix_flatten_of_sliced_array_0446():
    array = ak.highlevel.Array([[1, 2, 3], [], [4, 5], [6, 7, 8, 9]]).layout

    assert ak.operations.flatten(array[:-1], axis=1).to_list() == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert ak.operations.flatten(array[:-2], axis=1).to_list() == [1, 2, 3]
    assert ak.operations.flatten(array[:-1], axis=None).to_list() == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert ak.operations.flatten(array[:-2], axis=None).to_list() == [
        1,
        2,
        3,
    ]


def test_flatten_None_axis():
    array2 = ak.highlevel.Array(np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7))

    assert to_list(ak.operations.flatten(array2, axis=None)) == to_list(
        np.arange(2 * 3 * 5 * 7)
    )


def test_fix_corner_case_0585():
    array = ak.highlevel.Array([[1, 2, 3], [], [4, 5]]).layout

    assert ak.operations.flatten(array, axis=0).to_list() == [
        [1, 2, 3],
        [],
        [4, 5],
    ]
    assert ak.operations.flatten(array, axis=-2).to_list() == [
        [1, 2, 3],
        [],
        [4, 5],
    ]

    array = ak.highlevel.Array([1, 2, 3, 4, 5]).layout

    assert ak.operations.flatten(array, axis=0).to_list() == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert ak.operations.flatten(array, axis=-1).to_list() == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_flatten_allow_regulararray_size_zero_0590():

    empty = ak.contents.RegularArray(
        ak.highlevel.Array([[1, 2, 3], [], [4, 5]]).layout, 0, zeros_length=0
    )

    assert ak.operations.flatten(empty, axis=0).to_list() == []
    assert ak.operations.flatten(empty, axis=1).to_list() == []
    assert ak.operations.flatten(empty, axis=2).to_list() == []


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


def test_flatten_axis_none_0866():
    a1 = ak.operations.zip(
        {"a": [[1], [], [2, 3]], "b": [[4], [], [5, 6]]}, with_name="a1"
    )
    a2 = ak.operations.zip(
        {"a": [[7, 8], [9], []], "b": [[10, 11], [12], []]}, with_name="a2"
    )
    union = ak.operations.where([True, False, True], a1, a2)
    assert set(ak.operations.flatten(union, axis=None)) == {
        1,
        2,
        3,
        4,
        5,
        6,
        9,
        12,
    }


def test_0973():
    array = ak.operations.from_numpy(np.zeros((3, 3, 5)))
    flattened = ak.operations.flatten(array, axis=None)
    assert flattened.ndim == 1
