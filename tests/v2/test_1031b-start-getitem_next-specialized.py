# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2, v1v2_equal

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_NumpyArray():
    a = ak._v2.contents.RegularArray(
        v1_to_v2(ak.from_numpy(np.arange(2 * 3 * 5).reshape(-1, 5)).layout), 3
    )
    assert ak.to_list(a[1]) == [
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29],
    ]
    assert ak.to_list(a[1, -2]) == [20, 21, 22, 23, 24]
    assert a.typetracer[1, -2].form == a[1, -2].form
    assert a[1, -2, 2] == 22
    with pytest.raises(IndexError):
        a[1, -2, 2, 0]
    assert ak.to_list(a[1, -2, 2:]) == [22, 23, 24]
    assert a.typetracer[1, -2, 2:].form == a[1, -2, 2:].form
    with pytest.raises(IndexError):
        a[1, -2, 2:, 0]
    with pytest.raises(IndexError):
        a[1, -2, "hello"]
    with pytest.raises(IndexError):
        a[1, -2, ["hello", "there"]]
    assert ak.to_list(a[1, -2, np.newaxis, 2]) == [22]
    assert ak.to_list(a[1, -2, np.newaxis, np.newaxis, 2]) == [[22]]
    assert ak.to_list(a[1, -2, ...]) == [20, 21, 22, 23, 24]
    assert a.typetracer[1, -2, ...].form == a[1, -2, ...].form
    assert a.typetracer[1, ..., -2].form == a[1, ..., -2].form
    assert a[1, -2, ..., 2] == 22
    with pytest.raises(IndexError):
        a[1, -2, ..., 2, 2]

    b = ak.layout.RegularArray(
        ak.from_numpy(np.arange(2 * 3 * 5).reshape(-1, 5)).layout, 3
    )

    assert ak.to_list(b[1, -2, [3, 1, 1, 2]]) == [23, 21, 21, 22]
    assert ak.to_list(a[1, -2, [3, 1, 1, 2]]) == [23, 21, 21, 22]

    with pytest.raises(IndexError):
        a[1, -2, [3, 1, 1, 2], 2]


def test_RegularArray():
    old = ak.layout.RegularArray(
        ak.from_numpy(np.arange(2 * 3 * 5).reshape(-1, 5)).layout, 3
    )
    new = v1_to_v2(old)

    assert ak.to_list(old[1, 1:]) == [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
    assert ak.to_list(new[1, 1:]) == [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
    assert v1v2_equal(old[0, 1:], new[0, 1:])
    assert new.typetracer[1, 1:].form == new[1, 1:].form

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert ak.to_list(new[1, np.newaxis, -2]) == [[20, 21, 22, 23, 24]]
    assert ak.to_list(new[1, np.newaxis, np.newaxis, -2]) == [[[20, 21, 22, 23, 24]]]
    assert new.typetracer[1, np.newaxis, -2].form == new[1, np.newaxis, -2].form

    assert old.minmax_depth == (3, 3)
    assert new.minmax_depth == (3, 3)

    assert ak.to_list(old[1, ..., -2]) == [18, 23, 28]
    assert ak.to_list(new[1, ..., -2]) == [18, 23, 28]
    assert new.typetracer[1, ..., -2].form == new[1, ..., -2].form

    expectation = [
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
    ]
    assert (
        ak.to_list(
            old[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        ak.to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        new.typetracer[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )
    assert ak.to_list(new[[1, 0]]) == expectation

    assert ak.to_list(old[1, [2, 0]]) == [[25, 26, 27, 28, 29], [15, 16, 17, 18, 19]]
    assert ak.to_list(new[1, [2, 0]]) == [[25, 26, 27, 28, 29], [15, 16, 17, 18, 19]]


def test_RecordArray():
    old = ak.layout.RecordArray(
        [
            ak.layout.NumpyArray(
                np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], np.int64)
            ),
            ak.layout.NumpyArray(
                np.array(
                    [[0.0, 1.1, 2.2, 3.3, 4.4, 5.5], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]]
                )
            ),
        ],
        ["x", "y"],
    )

    new = v1_to_v2(old)

    assert ak.to_list(old[:, 3:]) == [
        {"x": [3, 4], "y": [3.3, 4.4, 5.5]},
        {"x": [3, 4], "y": [3.3, 4.4, 5.5]},
    ]
    assert ak.to_list(new[:, 3:]) == [
        {"x": [3, 4], "y": [3.3, 4.4, 5.5]},
        {"x": [3, 4], "y": [3.3, 4.4, 5.5]},
    ]
    assert new.typetracer[:, 3:].form == new[:, 3:].form

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert ak.to_list(new[1, np.newaxis]) == [
        {"x": [0, 1, 2, 3, 4], "y": [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]}
    ]
    assert ak.to_list(old[1, np.newaxis]) == [
        {"x": [0, 1, 2, 3, 4], "y": [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]}
    ]
    assert new.typetracer[1, np.newaxis].form == new[1, np.newaxis].form

    assert old.minmax_depth == (2, 2)
    assert new.minmax_depth == (2, 2)

    assert ak.to_list(old[0, ..., 0]) == {"x": 0, "y": 0.0}
    assert ak.to_list(new[0, ..., 0]) == {"x": 0, "y": 0.0}
    assert new.typetracer[0, ..., 0].array.form == new[0, ..., 0].array.form

    expectation = [
        {"x": [0, 1, 2, 3, 4], "y": [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]},
        {"x": [0, 1, 2, 3, 4], "y": [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]},
    ]
    assert (
        ak.to_list(
            old[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        ak.to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        new.typetracer[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )
    assert ak.to_list(new[[1, 0]]) == expectation

    assert ak.to_list(old[1, [1, 0]]) == [{"x": 1, "y": 1.1}, {"x": 0, "y": 0.0}]
    assert ak.to_list(new[1, [1, 0]]) == [{"x": 1, "y": 1.1}, {"x": 0, "y": 0.0}]


def test_UnmaskedArray():
    old = ak.layout.UnmaskedArray(
        ak.layout.NumpyArray(np.array([[0.0, 1.1, 2.2, 3.3], [0.0, 1.1, 2.2, 3.3]]))
    )
    new = v1_to_v2(old)

    assert ak.to_list(old[0, 1:]) == [1.1, 2.2, 3.3]
    assert ak.to_list(new[0, 1:]) == [1.1, 2.2, 3.3]
    assert v1v2_equal(old[0, 1:], new[0, 1:])
    assert new.typetracer[0, 1:].form == new[0, 1:].form

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert ak.to_list(new[1, np.newaxis, -2]) == [2.2]
    assert ak.to_list(new[1, np.newaxis, np.newaxis, -2]) == [[2.2]]
    assert new.typetracer[1, np.newaxis, -2].form == new[1, np.newaxis, -2].form

    assert old.minmax_depth == (2, 2)
    assert new.minmax_depth == (2, 2)

    assert ak.to_list(old[1, ..., -2]) == 2.2
    assert ak.to_list(new[1, ..., -2]) == 2.2

    expectation = [[0.0, 1.1, 2.2, 3.3], [0.0, 1.1, 2.2, 3.3]]
    assert (
        ak.to_list(
            old[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        ak.to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert ak.to_list(new[[1, 0]]) == expectation
    assert v1v2_equal(old[1, ...], new[1, ...])
    assert ak.to_list(old[1, [1, 0]]) == [1.1, 0.0]
    assert ak.to_list(new[1, [1, 0]]) == [1.1, 0.0]
    assert (
        new.typetracer[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )


def test_UnionArray():
    old = ak.layout.UnionArray8_64(
        ak.layout.Index8(np.array([1, 1], np.int8)),
        ak.layout.Index64(np.array([1, 0], np.int64)),
        [
            ak.layout.RegularArray(
                ak.from_numpy(np.arange(2 * 3 * 5).reshape(-1, 5)).layout, 3
            ),
            ak.layout.RegularArray(
                ak.from_numpy(np.arange(2 * 3 * 5).reshape(-1, 5)).layout, 3
            ),
        ],
    )
    new = v1_to_v2(old)
    assert new.typetracer[1, [1, 0]].form == new[1, [1, 0]].form

    assert ak.to_list(old[0, :]) == [
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29],
    ]
    assert ak.to_list(new[0, :]) == [
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29],
    ]
    assert v1v2_equal(old[0, :], new[0, :])

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert ak.to_list(new[0, np.newaxis]) == [
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
    ]
    assert ak.to_list(old[0, np.newaxis]) == [
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
    ]
    assert new.typetracer[0, np.newaxis].form == new[0, np.newaxis].form
    assert old.minmax_depth == (3, 3)
    assert new.minmax_depth == (3, 3)

    assert ak.to_list(old[1, ...]) == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
    ]
    assert ak.to_list(new[1, ...]) == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
    ]

    expectation = [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert (
        ak.to_list(
            old[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        ak.to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        new.typetracer[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )
    assert ak.to_list(old[[1, 0]]) == expectation
    assert ak.to_list(new[[1, 0]]) == expectation
    assert ak.to_list(old[1, [1, 0]]) == [[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]
    assert ak.to_list(new[1, [1, 0]]) == [[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]


def test_IndexedArray():
    old = ak.layout.IndexedArray64(
        ak.layout.Index64(np.array([1, 0], np.int64)),
        ak.layout.RegularArray(
            ak.from_numpy(np.arange(2 * 3 * 5).reshape(-1, 5)).layout, 3
        ),
    )
    new = v1_to_v2(old)
    assert v1v2_equal(old[1, 1:], new[1, 1:])

    assert ak.to_list(old[1, 1:]) == [[5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
    assert ak.to_list(new[1, 1:]) == [[5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
    assert new.typetracer[1, 1:].form == new[1, 1:].form

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert ak.to_list(new[0, np.newaxis]) == [
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
    ]
    assert ak.to_list(old[0, np.newaxis]) == [
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
    ]
    assert new.typetracer[0, np.newaxis].form == new[0, np.newaxis].form

    assert old.minmax_depth == (3, 3)
    assert new.minmax_depth == (3, 3)

    assert v1v2_equal(old[1, ...], new[1, ...])
    assert ak.to_list(old[1, ...]) == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
    ]
    assert ak.to_list(new[1, ...]) == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
    ]
    assert new[1, ...].form == new.typetracer[1, ...].form

    expectation = [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert (
        ak.to_list(
            old[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        ak.to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        new.typetracer[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )
    assert ak.to_list(old[[1, 0]]) == expectation
    assert ak.to_list(new[[1, 0]]) == expectation

    assert ak.to_list(old[1, [1, 0]]) == [[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]
    assert ak.to_list(new[1, [1, 0]]) == [[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]
    assert v1v2_equal(old[1, [1, 0]], new[1, [1, 0]])
    assert new.typetracer[1, [1, 0]].form == new[1, [1, 0]].form


def test_BitMaskedArray():
    old = ak.layout.BitMaskedArray(
        ak.layout.IndexU8(
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
        ak.layout.NumpyArray(
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

    new = v1_to_v2(old)

    assert v1v2_equal(old[:, :], new[:, :])

    assert ak.to_list(old[:, 5:]) == [
        [5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    ]

    assert ak.to_list(new[:, 5:]) == [
        [5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    ]
    assert new.typetracer[:, 5:].form == new[:, 5:].form

    assert v1v2_equal(old[:, 5:], new[:, 5:])

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert ak.to_list(new[1, np.newaxis]) == [
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    ]
    assert ak.to_list(old[1, np.newaxis]) == [
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    ]
    assert new.typetracer[1, np.newaxis].form == new[1, np.newaxis].form

    assert old.minmax_depth == (2, 2)
    assert new.minmax_depth == (2, 2)

    assert ak.to_list(old[0, ..., 0]) == 0.0
    assert ak.to_list(new[0, ..., 0]) == 0.0

    expectation = [
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    ]
    assert (
        ak.to_list(
            old[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        ak.to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        new.typetracer[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )
    assert ak.to_list(old[[1, 0]]) == expectation
    assert ak.to_list(new[[1, 0]]) == expectation

    assert ak.to_list(old[1, [1, 0]]) == [1.0, 0.0]
    assert ak.to_list(new[1, [1, 0]]) == [1.0, 0.0]
    assert v1v2_equal(old[1, [1, 0]], new[1, [1, 0]])
    assert new.typetracer[1, [1, 0]].form == new[1, [1, 0]].form


def test_ByteMaskedArray():
    old = ak.layout.ByteMaskedArray(
        ak.layout.Index8(np.array([1, 1, 1], np.int8)),
        ak.layout.NumpyArray(
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

    new = v1_to_v2(old)

    assert ak.to_list(old[:, 5:]) == [[6.6], [6.6], [6.6]]
    assert ak.to_list(new[:, 5:]) == [[6.6], [6.6], [6.6]]
    assert new.typetracer[:, 5:].form == new[:, 5:].form

    assert v1v2_equal(old[:, 5:], new[:, 5:])

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert ak.to_list(new[1, np.newaxis]) == [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6]]
    assert ak.to_list(old[1, np.newaxis]) == [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6]]
    assert new.typetracer[1, np.newaxis].form == new[1, np.newaxis].form

    assert old.minmax_depth == (2, 2)
    assert new.minmax_depth == (2, 2)

    assert ak.to_list(old[0, ..., 0]) == 1.1
    assert ak.to_list(new[0, ..., 0]) == 1.1

    expectation = [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]]
    assert (
        ak.to_list(
            old[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        ak.to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        new.typetracer[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )
    assert ak.to_list(new[[1, 0]]) == expectation

    assert ak.to_list(old[1, [1, 0]]) == [2.2, 1.1]
    assert ak.to_list(new[1, [1, 0]]) == [2.2, 1.1]


def test_IndexedOptionArray():
    old = ak.layout.IndexedOptionArray64(
        ak.layout.Index64(np.array([1, 1], np.int64)),
        ak.layout.NumpyArray(
            np.array([[1.1, 2.2, 3.3, 4.4, 5.5, 6.6], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]])
        ),
    )

    new = v1_to_v2(old)

    assert ak.to_list(old[:, 3:]) == [[4.4, 5.5, 6.6], [4.4, 5.5, 6.6]]
    assert ak.to_list(new[:, 3:]) == [[4.4, 5.5, 6.6], [4.4, 5.5, 6.6]]
    assert new.typetracer[:, 3:].form == new[:, 3:].form

    assert v1v2_equal(old[:, 5:], new[:, 5:])

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert ak.to_list(new[1, np.newaxis]) == [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6]]
    assert ak.to_list(old[1, np.newaxis]) == [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6]]
    assert new.typetracer[1, np.newaxis].form == new[1, np.newaxis].form

    assert old.minmax_depth == (2, 2)
    assert new.minmax_depth == (2, 2)

    assert ak.to_list(old[0, ..., 0]) == 1.1
    assert ak.to_list(new[0, ..., 0]) == 1.1

    expectation = [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]]
    assert (
        ak.to_list(
            old[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        ak.to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        new.typetracer[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )
    assert ak.to_list(new[[1, 0]]) == expectation

    assert ak.to_list(old[1, [1, 0]]) == [2.2, 1.1]
    assert ak.to_list(new[1, [1, 0]]) == [2.2, 1.1]
    assert new.typetracer[1, [1, 0]].form == new[1, [1, 0]].form


def test_ListArray():
    old = ak.layout.ListArray64(
        ak.layout.Index64(np.array([0, 100, 1], np.int64)),
        ak.layout.Index64(np.array([3, 100, 3, 200], np.int64)),
        ak.layout.NumpyArray(
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
    new = v1_to_v2(old)

    assert ak.to_list(old[0, :2]) == [
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    ]
    assert ak.to_list(new[0, :2]) == [
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    ]
    assert new.typetracer[0, :2].form == new[0, :2].form

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert ak.to_list(new[0, np.newaxis]) == [
        [
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        ]
    ]
    assert new.typetracer[0, np.newaxis].form == new[0, np.newaxis].form
    assert ak.to_list(old[0, np.newaxis]) == [
        [
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        ]
    ]

    assert old.minmax_depth == (3, 3)
    assert new.minmax_depth == (3, 3)

    assert ak.to_list(old[0, ...]) == [
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    ]
    assert ak.to_list(new[0, ...]) == [
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    ]
    assert new.typetracer[0, ...].form == new[0, ...].form

    expectation = [
        [],
        [
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        ],
    ]
    assert (
        ak.to_list(
            old[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        ak.to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        new.typetracer[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )
    assert ak.to_list(old[[1, 0]]) == expectation
    assert ak.to_list(new[[1, 0]]) == expectation

    assert ak.to_list(old[0, [1, 0]]) == [
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    ]
    assert ak.to_list(new[0, [1, 0]]) == [
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    ]
    assert new.typetracer[0, [1, 0]].form == new[0, [1, 0]].form


def test_ListOffsetArray_NumpyArray():
    old = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 1, 2, 3], np.int64)),
        ak.layout.NumpyArray(
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
    new = v1_to_v2(old)

    assert ak.to_list(old[0, 0:]) == [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6]]
    assert ak.to_list(new[0, 0:]) == [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6]]
    assert v1v2_equal(old[0, 1:], new[0, 1:])
    assert new.typetracer[0, 0:].form == new[0, 0:].form

    with pytest.raises(IndexError):
        new[1, "hello"]

    with pytest.raises(IndexError):
        new[1, ["hello", "there"]]

    assert ak.to_list(new[1, np.newaxis]) == [[[11.1, 22.2, 33.3, 44.4, 55.5, 66.6]]]
    assert ak.to_list(old[1, np.newaxis]) == [[[11.1, 22.2, 33.3, 44.4, 55.5, 66.6]]]
    assert new.typetracer[1, np.newaxis].form == new[1, np.newaxis].form

    assert old.minmax_depth == (3, 3)
    assert new.minmax_depth == (3, 3)

    assert ak.to_list(old[1, ...]) == [[11.1, 22.2, 33.3, 44.4, 55.5, 66.6]]
    assert ak.to_list(new[1, ...]) == [[11.1, 22.2, 33.3, 44.4, 55.5, 66.6]]
    assert new.typetracer[1, ...].form == new[1, ...].form

    expectation = [
        [[11.1, 22.2, 33.3, 44.4, 55.5, 66.6]],
        [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6]],
    ]
    assert (
        ak.to_list(
            old[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        ak.to_list(
            new[
                [1, 0],
            ]
        )
        == expectation
    )
    assert (
        new.typetracer[
            [1, 0],
        ].form
        == new[
            [1, 0],
        ].form
    )
    assert ak.to_list(old[[1, 0]]) == expectation
    assert ak.to_list(new[[1, 0]]) == expectation

    assert ak.to_list(old[0, [0, 0]]) == [
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    ]
    assert ak.to_list(new[0, [0, 0]]) == [
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    ]
    assert new.typetracer[0, [0, 0]].form == new[0, [0, 0]].form
