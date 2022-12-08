# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_listoffsetarray_localindex():
    v2_array = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    assert to_list(ak._do.local_index(v2_array, 0)) == [0, 1, 2, 3, 4]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, 1)) == [
        [0, 1, 2],
        [],
        [0, 1],
        [0],
        [0, 1, 2, 3],
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 1).form
        == ak._do.local_index(v2_array, 1).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [
        [0, 1, 2],
        [],
        [0, 1],
        [0],
        [0, 1, 2, 3],
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )
    assert to_list(ak._do.local_index(v2_array, -2)) == [0, 1, 2, 3, 4]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -2).form
        == ak._do.local_index(v2_array, -2).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -3)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 2)

    v2_array = ak.operations.from_iter(
        [[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]], [[6.6, 7.7, 8.8, 9.9]]],
        highlevel=False,
    )
    assert to_list(ak._do.local_index(v2_array, 0)) == [0, 1, 2, 3]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, 1)) == [[0, 1, 2], [], [0], [0]]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 1).form
        == ak._do.local_index(v2_array, 1).form
    )
    assert to_list(ak._do.local_index(v2_array, 2)) == [
        [[0, 1, 2], [], [0, 1]],
        [],
        [[0]],
        [[0, 1, 2, 3]],
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 2).form
        == ak._do.local_index(v2_array, 2).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [
        [[0, 1, 2], [], [0, 1]],
        [],
        [[0]],
        [[0, 1, 2, 3]],
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )
    assert to_list(ak._do.local_index(v2_array, -2)) == [[0, 1, 2], [], [0], [0]]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -2).form
        == ak._do.local_index(v2_array, -2).form
    )
    assert to_list(ak._do.local_index(v2_array, -3)) == [0, 1, 2, 3]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -3).form
        == ak._do.local_index(v2_array, -3).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -4)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 3)


def test_regulararray_localindex():
    v2_array = ak.operations.from_numpy(
        np.arange(2 * 3 * 5).reshape(2, 3, 5), regulararray=True, highlevel=False
    )
    assert to_list(ak._do.local_index(v2_array, 0)) == [0, 1]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, 1)) == [[0, 1, 2], [0, 1, 2]]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 1).form
        == ak._do.local_index(v2_array, 1).form
    )
    assert to_list(ak._do.local_index(v2_array, 2)) == [
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 2).form
        == ak._do.local_index(v2_array, 2).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )
    assert to_list(ak._do.local_index(v2_array, -2)) == [[0, 1, 2], [0, 1, 2]]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -2).form
        == ak._do.local_index(v2_array, -2).form
    )
    assert to_list(ak._do.local_index(v2_array, -3)) == [0, 1]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -3).form
        == ak._do.local_index(v2_array, -3).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -4)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 3)

    v2_array = ak.operations.from_numpy(
        np.arange(2 * 3 * 5 * 10).reshape(2, 3, 5, 10),
        regulararray=True,
        highlevel=False,
    )
    assert to_list(ak._do.local_index(v2_array, 0)) == [0, 1]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, 1)) == [[0, 1, 2], [0, 1, 2]]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 1).form
        == ak._do.local_index(v2_array, 1).form
    )
    assert to_list(ak._do.local_index(v2_array, 2)) == [
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 2).form
        == ak._do.local_index(v2_array, 2).form
    )
    assert to_list(ak._do.local_index(v2_array, 3)) == [
        [
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
        ],
        [
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
        ],
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 3).form
        == ak._do.local_index(v2_array, 3).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [
        [
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
        ],
        [
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
        ],
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )
    assert to_list(ak._do.local_index(v2_array, -2)) == [
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -2).form
        == ak._do.local_index(v2_array, -2).form
    )
    assert to_list(ak._do.local_index(v2_array, -3)) == [[0, 1, 2], [0, 1, 2]]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -3).form
        == ak._do.local_index(v2_array, -3).form
    )
    assert to_list(ak._do.local_index(v2_array, -4)) == [0, 1]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -4).form
        == ak._do.local_index(v2_array, -4).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -5)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 4)

    v2_array = ak.highlevel.Array(
        ak.contents.RegularArray(
            ak.highlevel.Array([[1, 2, 3], [], [4, 5]]).layout, 0, zeros_length=0
        )
    ).layout

    assert to_list(ak._do.local_index(v2_array, 0)) == []
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, 1)) == []
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 1).form
        == ak._do.local_index(v2_array, 1).form
    )
    assert to_list(ak._do.local_index(v2_array, 2)) == []
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 2).form
        == ak._do.local_index(v2_array, 2).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == []
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )
    assert to_list(ak._do.local_index(v2_array, -2)) == []
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -2).form
        == ak._do.local_index(v2_array, -2).form
    )
    assert to_list(ak._do.local_index(v2_array, -3)) == []
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -3).form
        == ak._do.local_index(v2_array, -3).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -4)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 3)


def test_bytemaskedarray_localindex():
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
    v2_array = ak.contents.ByteMaskedArray(mask, content, valid_when=False)

    assert to_list(v2_array) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert to_list(ak._do.local_index(v2_array, axis=0)) == [0, 1, 2, 3, 4]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), axis=0).form
        == ak._do.local_index(v2_array, axis=0).form
    )
    assert to_list(ak._do.local_index(v2_array, axis=-3)) == [0, 1, 2, 3, 4]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), axis=-3).form
        == ak._do.local_index(v2_array, axis=-3).form
    )
    assert to_list(ak._do.local_index(v2_array, axis=1)) == [
        [0, 1, 2],
        [],
        None,
        None,
        [0, 1],
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), axis=1).form
        == ak._do.local_index(v2_array, axis=1).form
    )
    assert to_list(ak._do.local_index(v2_array, axis=-2)) == [
        [0, 1, 2],
        [],
        None,
        None,
        [0, 1],
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), axis=-2).form
        == ak._do.local_index(v2_array, axis=-2).form
    )
    assert to_list(ak._do.local_index(v2_array, axis=2)) == [
        [[0, 1, 2], [], [0, 1]],
        [],
        None,
        None,
        [[], [0, 1, 2]],
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), axis=2).form
        == ak._do.local_index(v2_array, axis=2).form
    )
    assert to_list(ak._do.local_index(v2_array, axis=-1)) == [
        [[0, 1, 2], [], [0, 1]],
        [],
        None,
        None,
        [[], [0, 1, 2]],
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), axis=-1).form
        == ak._do.local_index(v2_array, axis=-1).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, axis=4)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, axis=-4)


def test_numpyarray_localindex():
    v2_array = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
    )
    assert to_list(ak._do.local_index(v2_array, axis=0)) == [0, 1, 2, 3]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), axis=0).form
        == ak._do.local_index(v2_array, axis=0).form
    )
    assert to_list(ak._do.local_index(v2_array, axis=-1)) == [0, 1, 2, 3]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), axis=-1).form
        == ak._do.local_index(v2_array, axis=-1).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, axis=1)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, axis=2)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, axis=-2)


def test_bitmaskedarray_localindex():
    v2_array = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
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
                    dtype=np.uint8,
                )
            )
        ),
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )

    assert to_list(ak._do.local_index(v2_array, 0)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ]
    assert to_list(ak._do.local_index(v2_array, -1)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -2)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 1)

    v2_array = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=False,
    )

    assert to_list(ak._do.local_index(v2_array, 0)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -2)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 1)

    v2_array = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=True,
    )
    assert to_list(ak._do.local_index(v2_array, 0)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -2)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 1)


def test_unmaskedarray_localindex():
    v2_array = ak.contents.unmaskedarray.UnmaskedArray(
        ak.contents.numpyarray.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
        )
    )
    assert to_list(ak._do.local_index(v2_array, 0)) == [0, 1, 2, 3]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [0, 1, 2, 3]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -2)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 1)


def test_unionarray_localindex():
    v2_array = ak.contents.unionarray.UnionArray(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak.from_iter(["1", "2", "3"], highlevel=False),
            ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )
    assert to_list(ak._do.local_index(v2_array, 0)) == [0, 1, 2, 3, 4, 5, 6]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [0, 1, 2, 3, 4, 5, 6]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -2)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 1)


def test_recordarray_localindex():
    v2_array = ak.contents.regulararray.RegularArray(
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        3,
    )
    assert to_list(ak._do.local_index(v2_array, 0)) == [0, 1]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, 1)) == [[0, 1, 2], [0, 1, 2]]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 1).form
        == ak._do.local_index(v2_array, 1).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [[0, 1, 2], [0, 1, 2]]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )
    assert to_list(ak._do.local_index(v2_array, -2)) == [0, 1]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -2).form
        == ak._do.local_index(v2_array, -2).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -3)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 2)

    v2_array = ak.contents.regulararray.RegularArray(
        ak.contents.recordarray.RecordArray(
            [ak.contents.emptyarray.EmptyArray()], ["nest"]
        ),
        0,
        zeros_length=10,
    )

    assert to_list(ak._do.local_index(v2_array, 0)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, 1)) == [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 1).form
        == ak._do.local_index(v2_array, 1).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )
    assert to_list(ak._do.local_index(v2_array, -2)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -2).form
        == ak._do.local_index(v2_array, -2).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -3)

    v2_array = ak.contents.listarray.ListArray(
        ak.index.Index(np.array([4, 100, 1])),
        ak.index.Index(np.array([7, 100, 3, 200])),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
                )
            ],
            ["nest"],
        ),
    )

    assert to_list(ak._do.local_index(v2_array, 0)) == [0, 1, 2]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, 1)) == [[0, 1, 2], [], [0, 1]]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 1).form
        == ak._do.local_index(v2_array, 1).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [[0, 1, 2], [], [0, 1]]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )
    assert to_list(ak._do.local_index(v2_array, -2)) == [0, 1, 2]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -2).form
        == ak._do.local_index(v2_array, -2).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -3)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 2)

    v2_array = ak.contents.listoffsetarray.ListOffsetArray(
        ak.index.Index(np.array([1, 4, 4, 6])),
        ak.contents.recordarray.RecordArray(
            [ak.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7])],
            ["nest"],
        ),
    )
    assert to_list(ak._do.local_index(v2_array, 0)) == [0, 1, 2]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, 1)) == [[0, 1, 2], [], [0, 1]]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 1).form
        == ak._do.local_index(v2_array, 1).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [[0, 1, 2], [], [0, 1]]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )
    assert to_list(ak._do.local_index(v2_array, -2)) == [0, 1, 2]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -2).form
        == ak._do.local_index(v2_array, -2).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -3)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 2)

    v2_array = ak.contents.indexedarray.IndexedArray(
        ak.index.Index(np.array([2, 2, 0, 1, 4, 5, 4])),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
    )
    assert to_list(ak._do.local_index(v2_array, 0)) == [0, 1, 2, 3, 4, 5, 6]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [0, 1, 2, 3, 4, 5, 6]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -2)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 1)

    v2_array = ak.contents.indexedoptionarray.IndexedOptionArray(
        ak.index.Index(np.array([2, 2, -1, 1, -1, 5, 4])),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
    )
    assert to_list(ak._do.local_index(v2_array, 0)) == [0, 1, 2, 3, 4, 5, 6]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [0, 1, 2, 3, 4, 5, 6]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -2)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 1)

    v2_array = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([1, 0, 1, 0, 1], dtype=np.int8)),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        valid_when=True,
    )

    assert to_list(ak._do.local_index(v2_array, 0)) == [0, 1, 2, 3, 4]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [0, 1, 2, 3, 4]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -2)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 1)

    v2_array = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([0, 1, 0, 1, 0], dtype=np.int8)),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        valid_when=False,
    )

    assert to_list(ak._do.local_index(v2_array, 0)) == [0, 1, 2, 3, 4]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [0, 1, 2, 3, 4]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -2)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 1)

    v2_array = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                    ]
                )
            )
        ),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array(
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
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )

    assert to_list(ak._do.local_index(v2_array, 0)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -2)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 1)

    v2_array = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array(
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
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=False,
        length=13,
        lsb_order=False,
    )

    assert to_list(ak._do.local_index(v2_array, 0)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -2)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 1)

    v2_array = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array(
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
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=True,
        length=13,
        lsb_order=True,
    )

    assert to_list(ak._do.local_index(v2_array, 0)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -2)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 1)

    v2_array = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array(
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
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=False,
        length=13,
        lsb_order=True,
    )

    assert to_list(ak._do.local_index(v2_array, 0)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -2)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 1)

    v2_array = ak.contents.unmaskedarray.UnmaskedArray(
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
                )
            ],
            ["nest"],
        )
    )

    assert to_list(ak._do.local_index(v2_array, 0)) == [0, 1, 2, 3]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [0, 1, 2, 3]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -2)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 1)

    v2_array = ak.contents.unionarray.UnionArray(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak.contents.recordarray.RecordArray(
                [ak.from_iter(["1", "2", "3"], highlevel=False)], ["nest"]
            ),
            ak.contents.recordarray.RecordArray(
                [
                    ak.contents.numpyarray.NumpyArray(
                        np.array([1.1, 2.2, 3.3, 4.4, 5.5])
                    )
                ],
                ["nest"],
            ),
        ],
    )

    assert to_list(ak._do.local_index(v2_array, 0)) == [0, 1, 2, 3, 4, 5, 6]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), 0).form
        == ak._do.local_index(v2_array, 0).form
    )
    assert to_list(ak._do.local_index(v2_array, -1)) == [0, 1, 2, 3, 4, 5, 6]
    assert (
        ak._do.local_index(v2_array.to_typetracer(), -1).form
        == ak._do.local_index(v2_array, -1).form
    )

    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, -2)
    with pytest.raises(IndexError):
        ak._do.local_index(v2_array, 1)
