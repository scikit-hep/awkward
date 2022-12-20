# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_concatenate_number():
    a1 = ak.highlevel.Array([[1, 2, 3], [], [4, 5]]).layout
    a2 = ak.highlevel.Array([[[1.1], [2.2, 3.3]], [[]], [[4.4], [5.5]]]).layout
    a3 = ak.highlevel.Array([[123], [223], [323]]).layout

    assert to_list(ak.operations.concatenate([a1, 999], axis=1)) == [
        [1, 2, 3, 999],
        [999],
        [4, 5, 999],
    ]

    assert to_list(ak.operations.concatenate([a2, 999], axis=2)) == [
        [[1.1, 999.0], [2.2, 3.3, 999.0]],
        [[999.0]],
        [[4.4, 999.0], [5.5, 999.0]],
    ]

    assert (
        str(
            ak.operations.type(
                ak.operations.concatenate(
                    [
                        ak.highlevel.Array([[1, 2, 3], [], [4, 5]]),
                        ak.highlevel.Array([[123], [223], [323]]),
                    ],
                    axis=1,
                )
            )
        )
        == "3 * var * int64"
    )

    assert to_list(ak.operations.concatenate([a1, a3], axis=1)) == [
        [1, 2, 3, 123],
        [223],
        [4, 5, 323],
    ]


def test_list_offset_array_concatenate():
    content_one = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    content_two = ak.contents.NumpyArray(
        np.array([999.999, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99])
    )
    offsets_one = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10], dtype=np.int64))
    offsets_two = ak.index.Index64(np.array([1, 3, 4, 4, 6, 9, 9, 10], dtype=np.int64))
    offsets_three = ak.index.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    offsets_four = ak.index.Index64(np.array([1, 3, 4, 4, 6, 7, 7], dtype=np.int64))

    one = ak.contents.ListOffsetArray(offsets_one, content_one)
    two = ak.contents.ListOffsetArray(offsets_two, content_two)
    three = ak.contents.ListOffsetArray(offsets_three, one)
    four = ak.contents.ListOffsetArray(offsets_four, two)

    padded_one = ak._do.pad_none(one, 7, 0)
    assert to_list(padded_one) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
        None,
        None,
    ]

    assert to_list(ak.operations.concatenate([one, two], 0)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
        [0.11, 0.22],
        [0.33],
        [],
        [0.44, 0.55],
        [0.66, 0.77, 0.88],
        [],
        [0.99],
    ]

    assert to_list(ak.operations.concatenate([padded_one, two], axis=1)) == [
        [0.0, 1.1, 2.2, 0.11, 0.22],
        [0.33],
        [3.3, 4.4],
        [5.5, 0.44, 0.55],
        [6.6, 7.7, 8.8, 9.9, 0.66, 0.77, 0.88],
        [],
        [0.99],
    ]

    with pytest.raises(ValueError):
        assert to_list(ak.operations.concatenate([one, two], 2))

    assert to_list(ak.operations.concatenate([three, four], 0)) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        [[5.5], [6.6, 7.7, 8.8, 9.9]],
        [[0.33], []],
        [[0.44, 0.55]],
        [],
        [[0.66, 0.77, 0.88], []],
        [[0.99]],
        [],
    ]

    padded = ak._do.pad_none(three, 6, 0)

    assert to_list(ak.operations.concatenate([padded, four], 1)) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [0.33], []],
        [[0.44, 0.55]],
        [[5.5], [6.6, 7.7, 8.8, 9.9]],
        [[0.66, 0.77, 0.88], []],
        [[0.99]],
        [],
    ]

    assert to_list(ak.operations.concatenate([four, four], 2)) == [
        [[0.33, 0.33], []],
        [[0.44, 0.55, 0.44, 0.55]],
        [],
        [[0.66, 0.77, 0.88, 0.66, 0.77, 0.88], []],
        [[0.99, 0.99]],
        [],
    ]


def test_list_array_concatenate():
    one = ak.highlevel.Array([[1, 2, 3], [], [4, 5]]).layout
    two = ak.highlevel.Array([[1.1, 2.2], [3.3, 4.4], [5.5]]).layout

    one = ak.contents.ListArray(one.starts, one.stops, one.content)
    two = ak.contents.ListArray(two.starts, two.stops, two.content)

    assert to_list(ak.operations.concatenate([one, two], 0)) == [
        [1, 2, 3],
        [],
        [4, 5],
        [1.1, 2.2],
        [3.3, 4.4],
        [5.5],
    ]
    assert to_list(ak.operations.concatenate([one, two], 1)) == [
        [1, 2, 3, 1.1, 2.2],
        [3.3, 4.4],
        [4, 5, 5.5],
    ]


def test_records_concatenate():
    one = ak.highlevel.Array(
        [{"x": 1, "y": [1]}, {"x": 2, "y": [1, 2]}, {"x": 3, "y": [1, 2, 3]}]
    ).layout
    two = ak.highlevel.Array([{"y": [], "x": 4}, {"y": [3, 2, 1], "x": 5}]).layout

    assert to_list(ak.operations.concatenate([one, two], 0)) == [
        {"x": 1, "y": [1]},
        {"x": 2, "y": [1, 2]},
        {"x": 3, "y": [1, 2, 3]},
        {"y": [], "x": 4},
        {"y": [3, 2, 1], "x": 5},
    ]
    with pytest.raises(ValueError):
        to_list(ak.operations.concatenate([one, two], 1))

    with pytest.raises(ValueError):
        to_list(ak.operations.concatenate([one, two], 2))


def test_indexed_array_concatenate():
    one = ak.highlevel.Array([[1, 2, 3], [None, 4], None, [None, 5]]).layout
    two = ak.highlevel.Array([6, 7, 8]).layout
    three = ak.highlevel.Array([[6.6], [7.7, 8.8]]).layout
    four = ak.highlevel.Array([[6.6], [7.7, 8.8], None, [9.9]]).layout

    assert to_list(ak.operations.concatenate([one, two], 0)) == [
        [1, 2, 3],
        [None, 4],
        None,
        [None, 5],
        6,
        7,
        8,
    ]

    with pytest.raises(ValueError):
        to_list(ak.operations.concatenate([one, three], 1))

    assert to_list(ak.operations.concatenate([one, four], 1)) == [
        [1, 2, 3, 6.6],
        [None, 4, 7.7, 8.8],
        [],
        [None, 5, 9.9],
    ]


def test_bytemasked_concatenate():
    one = ak.contents.ByteMaskedArray(
        ak.index.Index8([True, True, False, True, False, True]),
        ak.highlevel.Array([1, 2, 3, 4, 5, 6]).layout,
        valid_when=True,
    )
    two = ak.contents.ByteMaskedArray(
        ak.index.Index8([True, False, False, True, True]),
        ak.highlevel.Array([7, 99, 999, 8, 9]).layout,
        valid_when=True,
    )

    assert to_list(ak.operations.concatenate([one, two], 0)) == [
        1,
        2,
        None,
        4,
        None,
        6,
        7,
        None,
        None,
        8,
        9,
    ]

    with pytest.raises(ValueError):
        to_list(ak.operations.concatenate([one, two], 1))


def test_listoffsetarray_concatenate():
    content_one = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    offsets_one = ak.index.Index64(np.array([0, 3, 3, 5, 9]))
    one = ak.contents.ListOffsetArray(offsets_one, content_one)

    assert to_list(one) == [[1, 2, 3], [], [4, 5], [6, 7, 8, 9]]

    content_two = ak.contents.NumpyArray(np.array([100, 200, 300, 400, 500]))
    offsets_two = ak.index.Index64(np.array([0, 2, 4, 4, 5]))
    two = ak.contents.ListOffsetArray(offsets_two, content_two)

    assert to_list(two) == [[100, 200], [300, 400], [], [500]]
    assert to_list(ak.operations.concatenate([one, two], 0)) == [
        [1, 2, 3],
        [],
        [4, 5],
        [6, 7, 8, 9],
        [100, 200],
        [300, 400],
        [],
        [500],
    ]
    assert to_list(ak.operations.concatenate([one, two], 1)) == [
        [1, 2, 3, 100, 200],
        [300, 400],
        [4, 5],
        [6, 7, 8, 9, 500],
    ]


def test_numpyarray_concatenate_axis0():
    np1 = np.arange(2 * 7 * 5, dtype=np.float64).reshape(2, 7, 5)
    np2 = np.arange(3 * 7 * 5, dtype=np.int64).reshape(3, 7, 5)
    ak1 = ak.contents.NumpyArray(np1)
    ak2 = ak.contents.NumpyArray(np2)

    assert to_list(ak.operations.concatenate([np1, np2, np1, np2], 0)) == to_list(
        np.concatenate([np1, np2, np1, np2], 0)
    )
    assert to_list(ak.operations.concatenate([ak1, ak2], 0)) == to_list(
        np.concatenate([np.asarray(ak1), np.asarray(ak2)], 0)
    )
    assert to_list(np.concatenate([np.asarray(ak1), np.asarray(ak2)], 0)) == to_list(
        ak.operations.concatenate([np1, np2], 0)
    )


def test_numpyarray_concatenate():

    np1 = np.arange(2 * 7 * 5, dtype=np.float64).reshape(2, 7, 5)
    np2 = np.arange(2 * 7 * 5, dtype=np.int64).reshape(2, 7, 5)
    ak1 = ak.contents.NumpyArray(np1)
    ak2 = ak.contents.NumpyArray(np2)

    assert to_list(np.concatenate([np1, np2], 1)) == to_list(
        ak.operations.concatenate([ak1, ak2], 1)
    )
    assert to_list(np.concatenate([np2, np1], 1)) == to_list(
        ak.operations.concatenate([ak2, ak1], 1)
    )
    assert to_list(np.concatenate([np1, np2], 2)) == to_list(
        ak.operations.concatenate([ak1, ak2], 2)
    )
    assert to_list(np.concatenate([np2, np1], 2)) == to_list(
        ak.operations.concatenate([ak2, ak1], 2)
    )
    assert to_list(np.concatenate([np2, np1], -1)) == to_list(
        ak.operations.concatenate([ak2, ak1], -1)
    )


def test_numbers_and_records_concatenate():
    numbers = [
        ak.highlevel.Array(
            [
                [1.1, 2.2, 3.3],
                [],
                [4.4, 5.5],
                [6.6, 7.7, 8.8, 9.9],
            ]
        ).layout,
        ak.highlevel.Array(
            [
                [10, 20],
                [30],
                [],
                [40, 50, 60],
            ]
        ).layout,
        ak.highlevel.Array(
            np.array(
                [
                    [101, 102, 103],
                    [201, 202, 203],
                    [301, 302, 303],
                    [401, 402, 403],
                ]
            )
        ).layout,
    ]

    records = [
        ak.highlevel.Array(
            [
                [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}],
                [],
                [{"x": 2.2, "y": [1, 2]}],
                [{"x": 3.3, "y": [1, 2, 3]}, {"x": 4.4, "y": [1, 2, 3, 4]}],
            ]
        ).layout,
        ak.highlevel.Array(
            [
                [{"x": 0.0, "y": []}],
                [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}],
                [],
                [{"x": 3.3, "y": [1, 2, 3]}],
            ]
        ).layout,
    ]

    assert to_list(ak.operations.concatenate([numbers[0], records[0]], 1)) == [
        [1.1, 2.2, 3.3, {"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}],
        [],
        [4.4, 5.5, {"x": 2.2, "y": [1, 2]}],
        [6.6, 7.7, 8.8, 9.9, {"x": 3.3, "y": [1, 2, 3]}, {"x": 4.4, "y": [1, 2, 3, 4]}],
    ]

    assert to_list(ak.operations.concatenate([numbers[0], numbers[1]], axis=1)) == [
        [1.1, 2.2, 3.3, 10, 20],
        [30],
        [4.4, 5.5],
        [6.6, 7.7, 8.8, 9.9, 40, 50, 60],
    ]

    assert to_list(ak.operations.concatenate(numbers, axis=1)) == [
        [1.1, 2.2, 3.3, 10, 20, 101, 102, 103],
        [30, 201, 202, 203],
        [4.4, 5.5, 301, 302, 303],
        [6.6, 7.7, 8.8, 9.9, 40, 50, 60, 401, 402, 403],
    ]

    assert to_list(ak.operations.concatenate(records, axis=1)) == [
        [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 0.0, "y": []}],
        [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}],
        [{"x": 2.2, "y": [1, 2]}],
        [
            {"x": 3.3, "y": [1, 2, 3]},
            {"x": 4.4, "y": [1, 2, 3, 4]},
            {"x": 3.3, "y": [1, 2, 3]},
        ],
    ]

    with pytest.raises(ValueError) as err:
        to_list(ak.operations.concatenate([numbers, records], axis=1))
    assert "cannot broadcast" in str(err.value)


def test_broadcast_and_apply_levels():
    arrays = [
        ak.highlevel.Array(
            [[[0.0, 1.1, 2.2], []], [[3.3, 4.4]], [[5.5], [6.6, 7.7, 8.8, 9.9]]]
        ).layout,
        ak.highlevel.Array([[[10, 20], [30]], [[40]], [[50, 60, 70], [80, 90]]]).layout,
    ]
    # nothing is required to have the same length
    assert ak.operations.concatenate(arrays, axis=0).to_list() == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4]],
        [[5.5], [6.6, 7.7, 8.8, 9.9]],
        [[10, 20], [30]],
        [[40]],
        [[50, 60, 70], [80, 90]],
    ]
    # the outermost arrays are required to have the same length, but nothing deeper than that
    assert ak.operations.concatenate(arrays, axis=1).to_list() == [
        [[0.0, 1.1, 2.2], [], [10, 20], [30]],
        [[3.3, 4.4], [40]],
        [[5.5], [6.6, 7.7, 8.8, 9.9], [50, 60, 70], [80, 90]],
    ]
    # the outermost arrays and the first level are required to have the same length, but nothing deeper
    assert ak.operations.concatenate(arrays, axis=2).to_list() == [
        [[0.0, 1.1, 2.2, 10, 20], [30]],
        [[3.3, 4.4, 40]],
        [[5.5, 50, 60, 70], [6.6, 7.7, 8.8, 9.9, 80, 90]],
    ]


def test_negative_axis_concatenate():
    one = ak.highlevel.Array(
        [[[0.0, 1.1, 2.2], []], [[3.3, 4.4]], [[5.5], [6.6, 7.7, 8.8, 9.9]]]
    ).layout
    two = ak.highlevel.Array(
        [[[10, 20], [30]], [[40]], [[50, 60, 70], [80, 90]]]
    ).layout
    arrays = [one, two]

    assert ak.operations.concatenate(arrays, axis=-1).to_list() == [
        [[0.0, 1.1, 2.2, 10, 20], [30]],
        [[3.3, 4.4, 40]],
        [[5.5, 50, 60, 70], [6.6, 7.7, 8.8, 9.9, 80, 90]],
    ]

    assert ak.operations.concatenate(arrays, axis=-2).to_list() == [
        [[0.0, 1.1, 2.2], [], [10, 20], [30]],
        [[3.3, 4.4], [40]],
        [[5.5], [6.6, 7.7, 8.8, 9.9], [50, 60, 70], [80, 90]],
    ]

    assert ak.operations.concatenate(arrays, axis=-3).to_list() == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4]],
        [[5.5], [6.6, 7.7, 8.8, 9.9]],
        [[10, 20], [30]],
        [[40]],
        [[50, 60, 70], [80, 90]],
    ]


def test_even_more():
    dim1 = ak.highlevel.Array([1.1, 2.2, 3.3, 4.4, 5.5]).layout
    dim1a = ak.highlevel.Array([[1.1], [2.2], [3.3], [4.4], [5.5]]).layout
    dim1b = ak.highlevel.Array(np.array([[1.1], [2.2], [3.3], [4.4], [5.5]])).layout
    dim2 = ak.highlevel.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]]).layout
    dim3 = ak.highlevel.Array(
        [[[0, 1, 2], []], [[3, 4]], [], [[5], [6, 7, 8, 9]], []]
    ).layout

    assert ak.operations.concatenate([dim1, 999]).to_list() == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        999,
    ]
    assert ak.operations.concatenate([999, dim1]).to_list() == [
        999,
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert ak.operations.concatenate([dim1, dim2]).to_list() == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
    ]
    assert ak.operations.concatenate([dim2, dim1]).to_list() == [
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]

    with pytest.raises(ValueError):
        ak.operations.concatenate([dim1, 999], axis=1)

    assert ak.operations.concatenate([dim2, 999], axis=1).to_list() == [
        [0, 1, 2, 999],
        [999],
        [3, 4, 999],
        [5, 999],
        [6, 7, 8, 9, 999],
    ]

    assert ak.operations.concatenate([999, dim2], axis=1).to_list() == [
        [999, 0, 1, 2],
        [999],
        [999, 3, 4],
        [999, 5],
        [999, 6, 7, 8, 9],
    ]

    with pytest.raises(ValueError):
        ak.operations.concatenate([dim1, dim2], axis=1)
    with pytest.raises(ValueError):
        ak.operations.concatenate([dim2, dim1], axis=1)
    assert ak.operations.concatenate([dim1a, dim2], axis=1).to_list() == [
        [1.1, 0, 1, 2],
        [2.2],
        [3.3, 3, 4],
        [4.4, 5],
        [5.5, 6, 7, 8, 9],
    ]
    assert ak.operations.concatenate([dim2, dim1a], axis=1).to_list() == [
        [0, 1, 2, 1.1],
        [2.2],
        [3, 4, 3.3],
        [5, 4.4],
        [6, 7, 8, 9, 5.5],
    ]
    assert ak.operations.concatenate([dim1b, dim2], axis=1).to_list() == [
        [1.1, 0, 1, 2],
        [2.2],
        [3.3, 3, 4],
        [4.4, 5],
        [5.5, 6, 7, 8, 9],
    ]
    assert ak.operations.concatenate([dim2, dim1b], axis=1).to_list() == [
        [0, 1, 2, 1.1],
        [2.2],
        [3, 4, 3.3],
        [5, 4.4],
        [6, 7, 8, 9, 5.5],
    ]

    assert ak.operations.concatenate([123, dim1, dim2]).to_list() == [
        123,
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
    ]
    assert ak.operations.concatenate([123, dim2, dim1]).to_list() == [
        123,
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert ak.operations.concatenate([dim1, 123, dim2]).to_list() == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        123,
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
    ]
    assert ak.operations.concatenate([dim1, dim2, 123]).to_list() == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
        123,
    ]
    assert ak.operations.concatenate([dim2, 123, dim1]).to_list() == [
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
        123,
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert ak.operations.concatenate([dim2, dim1, 123]).to_list() == [
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        123,
    ]

    with pytest.raises(ValueError):
        ak.operations.concatenate([123, dim1, dim2], axis=1)
    with pytest.raises(ValueError):
        ak.operations.concatenate([123, dim2, dim1], axis=1)
    with pytest.raises(ValueError):
        ak.operations.concatenate([dim1, 123, dim2], axis=1)
    with pytest.raises(ValueError):
        ak.operations.concatenate([dim1, dim2, 123], axis=1)
    with pytest.raises(ValueError):
        ak.operations.concatenate([dim2, 123, dim1], axis=1)
    with pytest.raises(ValueError):
        ak.operations.concatenate([dim2, dim1, 123], axis=1)

    assert ak.operations.concatenate([123, dim1a, dim2], axis=1).to_list() == [
        [123, 1.1, 0, 1, 2],
        [123, 2.2],
        [123, 3.3, 3, 4],
        [123, 4.4, 5],
        [123, 5.5, 6, 7, 8, 9],
    ]
    assert ak.operations.concatenate([123, dim2, dim1a], axis=1).to_list() == [
        [123, 0, 1, 2, 1.1],
        [123, 2.2],
        [123, 3, 4, 3.3],
        [123, 5, 4.4],
        [123, 6, 7, 8, 9, 5.5],
    ]
    assert ak.operations.concatenate([dim1a, 123, dim2], axis=1).to_list() == [
        [1.1, 123, 0, 1, 2],
        [2.2, 123],
        [3.3, 123, 3, 4],
        [4.4, 123, 5],
        [5.5, 123, 6, 7, 8, 9],
    ]
    assert ak.operations.concatenate([dim1a, dim2, 123], axis=1).to_list() == [
        [1.1, 0, 1, 2, 123],
        [2.2, 123],
        [3.3, 3, 4, 123],
        [4.4, 5, 123],
        [5.5, 6, 7, 8, 9, 123],
    ]
    assert ak.operations.concatenate([dim2, 123, dim1a], axis=1).to_list() == [
        [0, 1, 2, 123, 1.1],
        [123, 2.2],
        [3, 4, 123, 3.3],
        [5, 123, 4.4],
        [6, 7, 8, 9, 123, 5.5],
    ]
    assert ak.operations.concatenate([dim2, dim1a, 123], axis=1).to_list() == [
        [0, 1, 2, 1.1, 123],
        [2.2, 123],
        [3, 4, 3.3, 123],
        [5, 4.4, 123],
        [6, 7, 8, 9, 5.5, 123],
    ]

    assert ak.operations.concatenate([123, dim1b, dim2], axis=1).to_list() == [
        [123, 1.1, 0, 1, 2],
        [123, 2.2],
        [123, 3.3, 3, 4],
        [123, 4.4, 5],
        [123, 5.5, 6, 7, 8, 9],
    ]
    assert ak.operations.concatenate([123, dim2, dim1b], axis=1).to_list() == [
        [123, 0, 1, 2, 1.1],
        [123, 2.2],
        [123, 3, 4, 3.3],
        [123, 5, 4.4],
        [123, 6, 7, 8, 9, 5.5],
    ]
    assert ak.operations.concatenate([dim1b, 123, dim2], axis=1).to_list() == [
        [1.1, 123, 0, 1, 2],
        [2.2, 123],
        [3.3, 123, 3, 4],
        [4.4, 123, 5],
        [5.5, 123, 6, 7, 8, 9],
    ]
    assert ak.operations.concatenate([dim1b, dim2, 123], axis=1).to_list() == [
        [1.1, 0, 1, 2, 123],
        [2.2, 123],
        [3.3, 3, 4, 123],
        [4.4, 5, 123],
        [5.5, 6, 7, 8, 9, 123],
    ]
    assert ak.operations.concatenate([dim2, 123, dim1b], axis=1).to_list() == [
        [0, 1, 2, 123, 1.1],
        [123, 2.2],
        [3, 4, 123, 3.3],
        [5, 123, 4.4],
        [6, 7, 8, 9, 123, 5.5],
    ]
    assert ak.operations.concatenate([dim2, dim1b, 123], axis=1).to_list() == [
        [0, 1, 2, 1.1, 123],
        [2.2, 123],
        [3, 4, 3.3, 123],
        [5, 4.4, 123],
        [6, 7, 8, 9, 5.5, 123],
    ]

    assert ak.operations.concatenate([dim3, 123]).to_list() == [
        [[0, 1, 2], []],
        [[3, 4]],
        [],
        [[5], [6, 7, 8, 9]],
        [],
        123,
    ]
    assert ak.operations.concatenate([123, dim3]).to_list() == [
        123,
        [[0, 1, 2], []],
        [[3, 4]],
        [],
        [[5], [6, 7, 8, 9]],
        [],
    ]

    assert ak.operations.concatenate([dim3, 123], axis=1).to_list() == [
        [[0, 1, 2], [], 123],
        [[3, 4], 123],
        [123],
        [[5], [6, 7, 8, 9], 123],
        [123],
    ]
    assert ak.operations.concatenate([123, dim3], axis=1).to_list() == [
        [123, [0, 1, 2], []],
        [123, [3, 4]],
        [123],
        [123, [5], [6, 7, 8, 9]],
        [123],
    ]

    with pytest.raises(ValueError):
        ak.operations.concatenate([dim3, dim1], axis=1)
    with pytest.raises(ValueError):
        ak.operations.concatenate([dim1, dim3], axis=1)

    assert ak.operations.concatenate([dim3, dim2], axis=1).to_list() == [
        [[0, 1, 2], [], 0, 1, 2],
        [[3, 4]],
        [3, 4],
        [[5], [6, 7, 8, 9], 5],
        [6, 7, 8, 9],
    ]
    assert ak.operations.concatenate([dim2, dim3], axis=1).to_list() == [
        [0, 1, 2, [0, 1, 2], []],
        [[3, 4]],
        [3, 4],
        [5, [5], [6, 7, 8, 9]],
        [6, 7, 8, 9],
    ]

    assert ak.operations.concatenate([dim3, 123], axis=2).to_list() == [
        [[0, 1, 2, 123], [123]],
        [[3, 4, 123]],
        [],
        [[5, 123], [6, 7, 8, 9, 123]],
        [],
    ]
    assert ak.operations.concatenate([123, dim3], axis=2).to_list() == [
        [[123, 0, 1, 2], [123]],
        [[123, 3, 4]],
        [],
        [[123, 5], [123, 6, 7, 8, 9]],
        [],
    ]

    assert ak.operations.concatenate([dim3, dim3], axis=2).to_list() == [
        [[0, 1, 2, 0, 1, 2], []],
        [[3, 4, 3, 4]],
        [],
        [[5, 5], [6, 7, 8, 9, 6, 7, 8, 9]],
        [],
    ]

    rec1 = ak.highlevel.Array(
        [
            {"x": [1, 2], "y": [1.1]},
            {"x": [], "y": [2.2, 3.3]},
            {"x": [3], "y": []},
            {"x": [5, 6, 7], "y": []},
            {"x": [8, 9], "y": [4.4, 5.5]},
        ]
    ).layout
    rec2 = ak.highlevel.Array(
        [
            {"x": [100], "y": [10, 20]},
            {"x": [200], "y": []},
            {"x": [300, 400], "y": [30]},
            {"x": [], "y": [40, 50]},
            {"x": [400, 500], "y": [60]},
        ]
    ).layout

    assert ak.operations.concatenate([rec1, rec2]).to_list() == [
        {"x": [1, 2], "y": [1.1]},
        {"x": [], "y": [2.2, 3.3]},
        {"x": [3], "y": []},
        {"x": [5, 6, 7], "y": []},
        {"x": [8, 9], "y": [4.4, 5.5]},
        {"x": [100], "y": [10, 20]},
        {"x": [200], "y": []},
        {"x": [300, 400], "y": [30]},
        {"x": [], "y": [40, 50]},
        {"x": [400, 500], "y": [60]},
    ]

    assert ak.operations.concatenate([rec1, rec2], axis=1).to_list() == [
        {"x": [1, 2, 100], "y": [1.1, 10, 20]},
        {"x": [200], "y": [2.2, 3.3]},
        {"x": [3, 300, 400], "y": [30]},
        {"x": [5, 6, 7], "y": [40, 50]},
        {"x": [8, 9, 400, 500], "y": [4.4, 5.5, 60]},
    ]
