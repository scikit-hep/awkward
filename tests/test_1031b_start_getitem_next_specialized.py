# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_NumpyArray():
    a = ak.contents.RegularArray(
        ak.operations.from_numpy(np.arange(2 * 3 * 5).reshape(-1, 5)).layout,
        3,
    )
    assert to_list(a[1]) == [
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29],
    ]
    assert to_list(a[1, -2]) == [20, 21, 22, 23, 24]
    assert a.to_typetracer()[1, -2].form == a[1, -2].form
    assert a[1, -2, 2] == 22
    with pytest.raises(IndexError):
        a[1, -2, 2, 0]
    assert to_list(a[1, -2, 2:]) == [22, 23, 24]
    assert a.to_typetracer()[1, -2, 2:].form == a[1, -2, 2:].form
    with pytest.raises(IndexError):
        a[1, -2, 2:, 0]
    with pytest.raises(IndexError):
        a[1, -2, "hello"]
    with pytest.raises(IndexError):
        a[1, -2, ["hello", "there"]]
    assert to_list(a[1, -2, np.newaxis, 2]) == [22]
    assert to_list(a[1, -2, np.newaxis, np.newaxis, 2]) == [[22]]
    assert to_list(a[1, -2, ...]) == [20, 21, 22, 23, 24]
    assert a.to_typetracer()[1, -2, ...].form == a[1, -2, ...].form
    assert a.to_typetracer()[1, ..., -2].form == a[1, ..., -2].form
    assert a[1, -2, ..., 2] == 22
    with pytest.raises(IndexError):
        a[1, -2, ..., 2, 2]

    assert to_list(a[1, -2, [3, 1, 1, 2]]) == [23, 21, 21, 22]

    with pytest.raises(IndexError):
        a[1, -2, [3, 1, 1, 2], 2]


def test_RegularArray():
    new = ak.contents.RegularArray(
        ak.operations.from_numpy(np.arange(2 * 3 * 5).reshape(-1, 5)).layout,
        3,
    )

    assert to_list(new[1, 1:]) == [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
    assert new.to_typetracer()[1, 1:].form == new[1, 1:].form

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert to_list(new[1, np.newaxis, -2]) == [[20, 21, 22, 23, 24]]
    assert to_list(new[1, np.newaxis, np.newaxis, -2]) == [[[20, 21, 22, 23, 24]]]
    assert new.to_typetracer()[1, np.newaxis, -2].form == new[1, np.newaxis, -2].form

    assert new.minmax_depth == (3, 3)

    assert to_list(new[1, ..., -2]) == [18, 23, 28]
    assert new.to_typetracer()[1, ..., -2].form == new[1, ..., -2].form

    expectation = [
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
    ]
    assert (
        to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        new.to_typetracer()[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )
    assert to_list(new[[1, 0]]) == expectation

    assert to_list(new[1, [2, 0]]) == [[25, 26, 27, 28, 29], [15, 16, 17, 18, 19]]

    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    assert (
        repr(array[[True, False, True]])
        == "<Array [[1.1, 2.2, 3.3], [4.4, 5.5]] type='2 * var * float64'>"
    )
    assert (
        repr(array[[True, False, True], 1]) == "<Array [2.2, 5.5] type='2 * float64'>"
    )


def test_RecordArray():
    new = ak.contents.RecordArray(
        [
            ak.contents.NumpyArray(
                np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], np.int64)
            ),
            ak.contents.NumpyArray(
                np.array(
                    [[0.0, 1.1, 2.2, 3.3, 4.4, 5.5], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]]
                )
            ),
        ],
        ["x", "y"],
    )

    assert to_list(new[:, 3:]) == [
        {"x": [3, 4], "y": [3.3, 4.4, 5.5]},
        {"x": [3, 4], "y": [3.3, 4.4, 5.5]},
    ]
    assert new.to_typetracer()[:, 3:].form == new[:, 3:].form

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert to_list(new[1, np.newaxis]) == [
        {"x": [0, 1, 2, 3, 4], "y": [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]}
    ]
    assert new.to_typetracer()[1, np.newaxis].form == new[1, np.newaxis].form

    assert new.minmax_depth == (2, 2)

    assert to_list(new[0, ..., 0]) == {"x": 0, "y": 0.0}
    assert new.to_typetracer()[0, ..., 0].array.form == new[0, ..., 0].array.form

    expectation = [
        {"x": [0, 1, 2, 3, 4], "y": [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]},
        {"x": [0, 1, 2, 3, 4], "y": [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]},
    ]
    assert (
        to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        new.to_typetracer()[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )
    assert to_list(new[[1, 0]]) == expectation

    assert to_list(new[1, [1, 0]]) == [{"x": 1, "y": 1.1}, {"x": 0, "y": 0.0}]


def test_UnmaskedArray():
    new = ak.contents.UnmaskedArray(
        ak.contents.NumpyArray(np.array([[0.0, 1.1, 2.2, 3.3], [0.0, 1.1, 2.2, 3.3]]))
    )

    assert to_list(new[0, 1:]) == [1.1, 2.2, 3.3]
    assert isinstance(new.to_typetracer()[0, 1:], ak._typetracer.MaybeNone)
    assert new.to_typetracer()[0, 1:].content.form == new[0, 1:].form

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert to_list(new[1, np.newaxis, -2]) == [2.2]
    assert to_list(new[1, np.newaxis, np.newaxis, -2]) == [[2.2]]
    assert new.to_typetracer()[1, np.newaxis, -2].form == new[1, np.newaxis, -2].form

    assert new.minmax_depth == (2, 2)

    assert to_list(new[1, ..., -2]) == 2.2

    expectation = [[0.0, 1.1, 2.2, 3.3], [0.0, 1.1, 2.2, 3.3]]
    assert (
        to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert to_list(new[[1, 0]]) == expectation
    assert to_list(new[1, [1, 0]]) == [1.1, 0.0]
    assert (
        new.to_typetracer()[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )


def test_UnionArray():
    new = ak.contents.UnionArray(
        ak.index.Index8(np.array([1, 1], np.int8)),
        ak.index.Index64(np.array([1, 0], np.int64)),
        [
            ak.contents.RegularArray(
                ak.operations.from_iter(["nothing"]).layout[1:],
                3,
            ),
            ak.contents.RegularArray(
                ak.operations.from_numpy(np.arange(2 * 3 * 5).reshape(-1, 5)).layout,
                3,
            ),
        ],
    )
    assert new.to_typetracer()[1, [1, 0]].form == new[1, [1, 0]].form

    assert to_list(new[0, :]) == [
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29],
    ]

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert to_list(new[0, np.newaxis]) == [
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
    ]
    assert new.to_typetracer()[0, np.newaxis].form == new[0, np.newaxis].form
    assert new.minmax_depth == (2, 3)

    assert to_list(new[1, ...]) == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
    ]

    expectation = [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert (
        to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        new.to_typetracer()[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )
    assert to_list(new[[1, 0]]) == expectation
    assert to_list(new[1, [1, 0]]) == [[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]


def test_IndexedArray():
    new = ak.contents.IndexedArray(
        ak.index.Index64(np.array([1, 0], np.int64)),
        ak.contents.RegularArray(
            ak.operations.from_numpy(np.arange(2 * 3 * 5).reshape(-1, 5)).layout,
            3,
        ),
    )

    assert to_list(new[1, 1:]) == [[5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
    assert new.to_typetracer()[1, 1:].form == new[1, 1:].form

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert to_list(new[0, np.newaxis]) == [
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
    ]
    assert new.to_typetracer()[0, np.newaxis].form == new[0, np.newaxis].form

    assert new.minmax_depth == (3, 3)

    assert to_list(new[1, ...]) == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
    ]
    assert new[1, ...].form == new.to_typetracer()[1, ...].form

    expectation = [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert (
        to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        new.to_typetracer()[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )
    assert to_list(new[[1, 0]]) == expectation

    assert to_list(new[1, [1, 0]]) == [[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]
    assert new.to_typetracer()[1, [1, 0]].form == new[1, [1, 0]].form


def test_BitMaskedArray():
    new = ak.contents.BitMaskedArray(
        ak.index.IndexU8(
            np.packbits(
                np.array(
                    [
                        [
                            1,
                            1,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                        ],
                        [
                            1,
                            1,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                        ],
                    ],
                    np.uint8,
                )
            )
        ),
        ak.contents.NumpyArray(
            np.array(
                [
                    [
                        0.0,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        1.1,
                        2.2,
                        3.3,
                        4.4,
                        5.5,
                        6.6,
                    ],
                    [
                        0.0,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        1.1,
                        2.2,
                        3.3,
                        4.4,
                        5.5,
                        6.6,
                    ],
                ]
            )
        ),
        valid_when=True,
        length=2,
        lsb_order=False,
    )

    assert to_list(new[:, 5:]) == [
        [5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    ]
    assert new.to_typetracer()[:, 5:].form == new[:, 5:].form

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert to_list(new[1, np.newaxis]) == [
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    ]
    assert new.to_typetracer()[1, np.newaxis].form == new[1, np.newaxis].form

    assert new.minmax_depth == (2, 2)

    assert to_list(new[0, ..., 0]) == 0.0

    expectation = [
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    ]
    assert (
        to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        new.to_typetracer()[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )
    assert to_list(new[[1, 0]]) == expectation

    assert to_list(new[1, [1, 0]]) == [1.0, 0.0]
    assert new.to_typetracer()[1, [1, 0]].form == new[1, [1, 0]].form


def test_ByteMaskedArray():
    new = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([1, 1, 1], np.int8)),
        ak.contents.NumpyArray(
            np.array(
                [
                    [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
                    [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
                    [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
                ]
            )
        ),
        valid_when=True,
    )

    assert to_list(new[:, 5:]) == [[6.6], [6.6], [6.6]]
    assert new.to_typetracer()[:, 5:].form == new[:, 5:].form

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert to_list(new[1, np.newaxis]) == [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6]]
    assert new.to_typetracer()[1, np.newaxis].form == new[1, np.newaxis].form

    assert new.minmax_depth == (2, 2)

    assert to_list(new[0, ..., 0]) == 1.1

    expectation = [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]]
    assert (
        to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        new.to_typetracer()[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )
    assert to_list(new[[1, 0]]) == expectation

    assert to_list(new[1, [1, 0]]) == [2.2, 1.1]


def test_IndexedOptionArray():
    new = ak.contents.IndexedOptionArray(
        ak.index.Index64(np.array([1, 1], np.int64)),
        ak.contents.NumpyArray(
            np.array([[1.1, 2.2, 3.3, 4.4, 5.5, 6.6], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]])
        ),
    )

    assert to_list(new[:, 3:]) == [[4.4, 5.5, 6.6], [4.4, 5.5, 6.6]]
    assert new.to_typetracer()[:, 3:].form == new[:, 3:].form

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert to_list(new[1, np.newaxis]) == [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6]]
    assert new.to_typetracer()[1, np.newaxis].form == new[1, np.newaxis].form

    assert new.minmax_depth == (2, 2)

    assert to_list(new[0, ..., 0]) == 1.1

    expectation = [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]]
    assert (
        to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        new.to_typetracer()[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )
    assert to_list(new[[1, 0]]) == expectation

    assert to_list(new[1, [1, 0]]) == [2.2, 1.1]
    assert new.to_typetracer()[1, [1, 0]].form == new[1, [1, 0]].form


def test_ListArray():
    new = ak.contents.ListArray(
        ak.index.Index64(np.array([0, 100, 1], np.int64)),
        ak.index.Index64(np.array([3, 100, 3, 200], np.int64)),
        ak.contents.NumpyArray(
            np.array(
                [
                    [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
                    [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
                    [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
                    [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
                    [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
                ]
            )
        ),
    )

    assert to_list(new[0, :2]) == [
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    ]
    assert new.to_typetracer()[0, :2].form == new[0, :2].form

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert to_list(new[0, np.newaxis]) == [
        [
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        ]
    ]
    assert new.to_typetracer()[0, np.newaxis].form == new[0, np.newaxis].form

    assert new.minmax_depth == (3, 3)

    assert to_list(new[0, ...]) == [
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    ]
    assert new.to_typetracer()[0, ...].form == new[0, ...].form

    expectation = [
        [],
        [
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        ],
    ]
    assert (
        to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        new.to_typetracer()[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )
    assert to_list(new[[1, 0]]) == expectation

    assert to_list(new[0, [1, 0]]) == [
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    ]
    assert new.to_typetracer()[0, [1, 0]].form == new[0, [1, 0]].form


def test_ListOffsetArray_NumpyArray():
    new = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 2, 3], np.int64)),
        ak.contents.NumpyArray(
            np.array(
                [
                    [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
                    [11.1, 22.2, 33.3, 44.4, 55.5, 66.6],
                    [21.1, 22.2, 23.3, 24.4, 25.5, 26.6],
                    [31.1, 32.2, 33.3, 34.4, 35.5, 36.6],
                    [41.1, 42.2, 43.3, 44.4, 45.5, 46.6],
                ]
            )
        ),
    )

    assert to_list(new[0, 0:]) == [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6]]
    assert new.to_typetracer()[0, 0:].form == new[0, 0:].form

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert to_list(new[1, np.newaxis]) == [[[11.1, 22.2, 33.3, 44.4, 55.5, 66.6]]]
    assert new.to_typetracer()[1, np.newaxis].form == new[1, np.newaxis].form

    assert new.minmax_depth == (3, 3)

    assert to_list(new[1, ...]) == [[11.1, 22.2, 33.3, 44.4, 55.5, 66.6]]
    assert new.to_typetracer()[1, ...].form == new[1, ...].form

    expectation = [
        [[11.1, 22.2, 33.3, 44.4, 55.5, 66.6]],
        [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6]],
    ]
    assert (
        to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        new.to_typetracer()[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )
    assert to_list(new[[1, 0]]) == expectation

    assert to_list(new[0, [0, 0]]) == [
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    ]
    assert new.to_typetracer()[0, [0, 0]].form == new[0, [0, 0]].form
