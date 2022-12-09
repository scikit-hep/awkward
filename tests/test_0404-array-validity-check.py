# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test_BitMaskedArray():
    content = ak.contents.NumpyArray(np.arange(13))
    mask = ak.index.IndexU8(np.array([58, 59], dtype=np.uint8))
    array = ak.contents.BitMaskedArray(
        mask, content, valid_when=False, length=13, lsb_order=False
    )
    assert np.asarray(array.mask_as_bool(valid_when=True)).tolist() == [
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert np.asarray(array.to_ByteMaskedArray().mask).tolist() == [
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
    ]
    assert np.asarray(array.to_IndexedOptionArray64().index).tolist() == [
        0,
        1,
        -1,
        -1,
        -1,
        5,
        -1,
        7,
        8,
        9,
        -1,
        -1,
        -1,
    ]
    assert to_list(array) == [
        0,
        1,
        None,
        None,
        None,
        5,
        None,
        7,
        8,
        9,
        None,
        None,
        None,
    ]
    assert ak._do.is_unique(array, axis=None) is True
    assert to_list(ak._do.unique(array, axis=None)) == [0, 1, 5, 7, 8, 9, None]
    assert to_list(ak._do.unique(array, axis=-1)) == [0, 1, 5, 7, 8, 9, None]


def test_ByteMaskedArray_0():
    content = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.index.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.contents.ByteMaskedArray(mask, content, valid_when=False)
    assert to_list(array) == [[0.0, 1.1, 2.2], [], None, None, [6.6, 7.7, 8.8, 9.9]]
    assert ak._do.is_unique(array, axis=None) is True
    assert to_list(ak._do.unique(array, axis=None)) == [
        0.0,
        1.1,
        2.2,
        6.6,
        7.7,
        8.8,
        9.9,
        None,
    ]
    assert to_list(ak._do.unique(array, axis=-1)) == [
        [0.0, 1.1, 2.2],
        [],
        [6.6, 7.7, 8.8, 9.9],
        None,
        None,
    ]


def test_ByteMaskedArray_1():
    content = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [], [1.1, 2.2], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.index.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.contents.ByteMaskedArray(mask, content, valid_when=False)

    assert to_list(array) == [[0.0, 1.1, 2.2], [], None, None, [6.6, 7.7, 8.8, 9.9]]
    assert ak._do.is_unique(array, axis=None) is True
    assert to_list(ak._do.unique(array, axis=None)) == [
        0.0,
        1.1,
        2.2,
        6.6,
        7.7,
        8.8,
        9.9,
        None,
    ]
    assert to_list(ak._do.unique(array, axis=-1)) == [
        [0.0, 1.1, 2.2],
        [],
        [6.6, 7.7, 8.8, 9.9],
        None,
        None,
    ]


def test_ByteMaskedArray_2():
    content = ak.operations.from_iter(
        [[1.1, 1.1, 2.2], [], [1.1, 2.2], [5.5], [1.1, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.index.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.contents.ByteMaskedArray(mask, content, valid_when=False)

    assert to_list(array) == [[1.1, 1.1, 2.2], [], None, None, [1.1, 7.7, 8.8, 9.9]]
    assert ak._do.is_unique(array, axis=None) is False
    assert to_list(ak._do.unique(array, axis=None)) == [1.1, 2.2, 7.7, 8.8, 9.9, None]
    assert to_list(ak._do.unique(array, axis=-1)) == [
        [1.1, 2.2],
        [],
        [1.1, 7.7, 8.8, 9.9],
        None,
        None,
    ]


def test_UnmaskedArray():
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)
    )
    array = ak.contents.UnmaskedArray(content)

    assert to_list(array) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert ak._do.is_unique(array, axis=None) is True
    assert to_list(ak._do.unique(array, axis=None)) == [1.1, 2.2, 3.3, 4.4, 5.5]


def test_subranges_equal():
    array = ak.contents.NumpyArray(
        np.array(
            [
                [3.3, 2.2, 5.5, 1.1, 4.4],
                [4.4, 2.2, 1.1, 3.3, 5.5],
                [2.2, 1.1, 4.4, 3.3, 5.5],
            ]
        )  # ,
        #     highlevel=False,
    )

    starts = ak.index.Index64(np.array([0, 5, 10]))
    stops = ak.index.Index64(np.array([5, 10, 15]))

    result = ak.sort(array, axis=-1, highlevel=False).content._subranges_equal(
        starts, stops, 15
    )
    assert result is True

    starts = ak.index.Index64(np.array([0, 7]))
    stops = ak.index.Index64(np.array([7, 15]))

    assert (
        ak.sort(array, axis=-1, highlevel=False).content._subranges_equal(
            starts, stops, 15
        )
        is False
    )

    starts = ak.index.Index64(np.array([0]))
    stops = ak.index.Index64(np.array([15]))

    assert (
        ak.sort(array, axis=-1, highlevel=False).content._subranges_equal(
            starts, stops, 15
        )
        is False
    )

    # FIXME:
    # starts = ak.index.Index64(np.array([0]))
    # stops = ak.index.Index64(np.array([5]))
    #
    # assert array.sort(axis=-1).content._subranges_equal(starts, stops, 15) is True


def test_categorical():
    array = ak.highlevel.Array(["1chchc", "1chchc", "2sss", "3", "4", "5"])
    categorical = ak.operations.ak_to_categorical.to_categorical(array)

    assert ak.operations.is_valid(categorical) is True
    assert ak._do.is_unique(categorical.layout) is False


def test_NumpyArray():
    array = ak.contents.NumpyArray(np.array([5, 6, 1, 3, 4, 5]))

    assert ak._do.is_unique(array) is False
    assert to_list(ak._do.unique(array)) == [1, 3, 4, 5, 6]


def test_2d():
    nparray = np.array(
        [
            [3.3, 2.2, 5.5, 1.1, 4.4],
            [4.4, 2.2, 1.1, 3.3, 5.5],
            [2.2, 1.1, 4.4, 3.3, 5.5],
        ]
    )

    assert to_list(np.unique(nparray)) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert to_list(np.unique(nparray, axis=0)) == [
        [2.2, 1.1, 4.4, 3.3, 5.5],
        [3.3, 2.2, 5.5, 1.1, 4.4],
        [4.4, 2.2, 1.1, 3.3, 5.5],
    ]
    assert to_list(np.unique(nparray, axis=1)) == [
        [1.1, 2.2, 3.3, 4.4, 5.5],
        [3.3, 2.2, 4.4, 5.5, 1.1],
        [3.3, 1.1, 2.2, 5.5, 4.4],
    ]
    assert to_list(np.unique(nparray, axis=-1)) == [
        [1.1, 2.2, 3.3, 4.4, 5.5],
        [3.3, 2.2, 4.4, 5.5, 1.1],
        [3.3, 1.1, 2.2, 5.5, 4.4],
    ]
    assert to_list(np.unique(nparray, axis=-2)) == [
        [2.2, 1.1, 4.4, 3.3, 5.5],
        [3.3, 2.2, 5.5, 1.1, 4.4],
        [4.4, 2.2, 1.1, 3.3, 5.5],
    ]

    array = ak.contents.NumpyArray(
        np.array(
            [
                [3.3, 2.2, 5.5, 1.1, 4.4],
                [4.4, 2.2, 1.1, 3.3, 5.5],
                [2.2, 1.1, 4.4, 3.3, 5.5],
            ]
        )
    )

    assert (
        ak._do.is_unique(
            array,
        )
        is False
    )
    assert ak._do.is_unique(array, axis=-1) is True

    assert to_list(ak._do.unique(array)) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert to_list(ak._do.unique(array, axis=-1)) == [
        [1.1, 2.2, 3.3, 4.4, 5.5],
        [1.1, 2.2, 3.3, 4.4, 5.5],
        [1.1, 2.2, 3.3, 4.4, 5.5],
    ]


def test_2d_in_axis():
    array = ak.contents.NumpyArray(
        np.array(
            [
                [3.3, 2.2, 5.5, 1.1, 4.4, 1.1],
                [4.4, 2.2, 1.1, 3.3, 5.5, 3.3],
                [2.2, 2.2, 1.1, 4.4, 3.3, 5.5],
            ]
        )
    )
    assert ak._do.is_unique(array) is False
    assert to_list(ak._do.unique(array)) == [1.1, 2.2, 3.3, 4.4, 5.5]

    assert to_list(ak.sort(array, axis=-1, highlevel=False)) == [
        [1.1, 1.1, 2.2, 3.3, 4.4, 5.5],
        [1.1, 2.2, 3.3, 3.3, 4.4, 5.5],
        [1.1, 2.2, 2.2, 3.3, 4.4, 5.5],
    ]
    assert to_list(ak._do.unique(array, axis=-1)) == [
        [1.1, 2.2, 3.3, 4.4, 5.5],
        [1.1, 2.2, 3.3, 4.4, 5.5],
        [1.1, 2.2, 3.3, 4.4, 5.5],
    ]


def test_3d():
    array = ak.contents.NumpyArray(
        np.array(
            [
                # axis 2:    0       1       2       3       4         # axis 1:
                [
                    [1.1, 2.2, 3.3, 4.4, 5.5],  # 0
                    [6.6, 7.7, 8.8, 9.9, 10.10],  # 1
                    [11.11, 12.12, 13.13, 14.14, 15.15],
                ],  # 2
                [
                    [-1.1, -2.2, -3.3, -4.4, -5.5],  # 3
                    [-6.6, -7.7, -8.8, -9.9, -10.1],  # 4
                    [-11.11, -12.12, -13.13, -14.14, -15.15],
                ],
            ]
        )
    )  # 5

    assert ak._do.is_unique(array) is True

    assert to_list(ak._do.unique(array)) == [
        -15.15,
        -14.14,
        -13.13,
        -12.12,
        -11.11,
        -10.1,
        -9.9,
        -8.8,
        -7.7,
        -6.6,
        -5.5,
        -4.4,
        -3.3,
        -2.2,
        -1.1,
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        6.6,
        7.7,
        8.8,
        9.9,
        10.1,
        11.11,
        12.12,
        13.13,
        14.14,
        15.15,
    ]

    assert to_list(ak._do.unique(array, axis=-1)) == [
        [
            [1.1, 2.2, 3.3, 4.4, 5.5],
            [6.6, 7.7, 8.8, 9.9, 10.1],
            [11.11, 12.12, 13.13, 14.14, 15.15],
        ],
        [
            [-5.5, -4.4, -3.3, -2.2, -1.1],
            [-10.1, -9.9, -8.8, -7.7, -6.6],
            [-15.15, -14.14, -13.13, -12.12, -11.11],
        ],
    ]


def test_3d_non_unique():
    np_array = np.array(
        [
            [
                [1.1, 13.13, 13.13, 4.4, 5.5],
                [13.13, 7.7, 13.13, 13.13, 10.10],
                [13.13, 12.12, 13.13, 13.13, 13.13],
            ],
            [
                [-1.1, -2.2, 13.13, -4.4, -5.5],
                [13.13, -7.7, 13.13, -9.9, -10.1],
                [-12.12, -12.12, -13.13, 13.13, -15.15],
            ],
        ]
    )

    array = ak.contents.NumpyArray(np_array)

    assert to_list(ak._do.unique(array)) == to_list(np.unique(np_array))
    assert to_list(ak._do.unique(array, axis=-1)) == [
        [[1.1, 4.4, 5.5, 13.13], [7.7, 10.1, 13.13], [12.12, 13.13]],
        [
            [-5.5, -4.4, -2.2, -1.1, 13.13],
            [-10.1, -9.9, -7.7, 13.13],
            [-15.15, -13.13, -12.12, 13.13],
        ],
    ]


def test_ListOffsetArray():
    array = ak.operations.from_iter(
        ["one", "two", "three", "four", "five"], highlevel=False
    )

    assert to_list(ak.sort(array, 0, highlevel=False)) == [
        "five",
        "four",
        "one",
        "three",
        "two",
    ]
    assert ak._do.is_unique(array) is True
    assert to_list(ak._do.unique(array, axis=-1)) == [
        "five",
        "four",
        "one",
        "three",
        "two",
    ]

    array2 = ak.operations.from_iter(
        ["one", "two", "one", "four", "two"], highlevel=False
    )
    assert to_list(ak.sort(array2, 0, highlevel=False)) == [
        "four",
        "one",
        "one",
        "two",
        "two",
    ]
    assert ak._do.is_unique(array2) is False
    assert to_list(ak._do.unique(array2, axis=-1)) == [
        "four",
        "one",
        "two",
    ]

    content = ak.contents.NumpyArray(
        np.array([3.3, 1.1, 2.2, 0.0, 4.4, 9.9, 6.6, 7.7, 8.8, 5.5])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    assert to_list(listoffsetarray) == [
        [3.3, 1.1, 2.2],
        [],
        [0.0, 4.4],
        [9.9],
        [6.6, 7.7, 8.8, 5.5],
        [],
    ]
    assert ak._do.is_unique(listoffsetarray) is True
    assert to_list(ak._do.unique(listoffsetarray, axis=-1)) == [
        [1.1, 2.2, 3.3],
        [],
        [0.0, 4.4],
        [9.9],
        [5.5, 6.6, 7.7, 8.8],
        [],
    ]
    assert to_list(ak._do.unique(listoffsetarray)) == [
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

    content = ak.contents.NumpyArray(
        np.array([3.3, 1.1, 2.2, 0.0, 4.4, 9.9, 2.2, 3.3, 1.1, 5.5])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 9, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    assert to_list(listoffsetarray) == [
        [3.3, 1.1, 2.2],
        [],
        [0.0, 4.4],
        [9.9],
        [2.2, 3.3, 1.1],
        [5.5],
    ]
    assert ak._do.is_unique(listoffsetarray) is False
    assert to_list(ak._do.unique(listoffsetarray, axis=-1)) == [
        [1.1, 2.2, 3.3],
        [],
        [0.0, 4.4],
        [9.9],
        [1.1, 2.2, 3.3],
        [5.5],
    ]

    content1 = ak.contents.NumpyArray(
        np.array([3.3, 1.1, 2.2, 2.2, 3.3, 1.1, 0.0, 4.4, 9.9, 5.5])
    )
    offsets1 = ak.index.Index64(np.array([0, 3, 6, 6, 8, 9, 10]))
    listoffsetarray1 = ak.contents.ListOffsetArray(offsets1, content1)
    assert to_list(listoffsetarray1) == [
        [3.3, 1.1, 2.2],
        [2.2, 3.3, 1.1],
        [],
        [0.0, 4.4],
        [9.9],
        [5.5],
    ]
    assert ak._do.is_unique(listoffsetarray1) is False
    assert to_list(ak._do.unique(listoffsetarray1, axis=-1)) == [
        [1.1, 2.2, 3.3],
        [1.1, 2.2, 3.3],
        [],
        [0.0, 4.4],
        [9.9],
        [5.5],
    ]

    content2 = ak.contents.NumpyArray(
        np.array([3.3, 1.1, 2.2, 2.2, 3.3, 1.1, 0.0, 4.4, 9.9, 5.5, 5.5])
    )
    offsets2 = ak.index.Index64(np.array([0, 3, 6, 6, 8, 9, 10, 11]))
    listoffsetarray2 = ak.contents.ListOffsetArray(offsets2, content2)
    assert to_list(listoffsetarray2) == [
        [3.3, 1.1, 2.2],
        [2.2, 3.3, 1.1],
        [],
        [0.0, 4.4],
        [9.9],
        [5.5],
        [5.5],
    ]
    assert ak._do.is_unique(listoffsetarray2) is False
    assert to_list(ak._do.unique(listoffsetarray2, axis=-1)) == [
        [1.1, 2.2, 3.3],
        [1.1, 2.2, 3.3],
        [],
        [0.0, 4.4],
        [9.9],
        [5.5],
        [5.5],
    ]

    content2 = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 1.1, 5.5, 6.6, 7.7, 2.2, 9.9])
    )
    offsets2 = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray2 = ak.contents.ListOffsetArray(offsets2, content2)
    assert to_list(listoffsetarray2) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 1.1],
        [5.5],
        [6.6, 7.7, 2.2, 9.9],
        [],
    ]
    assert to_list(ak.sort(listoffsetarray2, highlevel=False)) == [
        [0.0, 1.1, 2.2],
        [],
        [1.1, 3.3],
        [5.5],
        [2.2, 6.6, 7.7, 9.9],
        [],
    ]
    assert ak._do.is_unique(listoffsetarray2, axis=-1) is True
    assert to_list(ak._do.unique(listoffsetarray2, axis=-1)) == [
        [0.0, 1.1, 2.2],
        [],
        [1.1, 3.3],
        [5.5],
        [2.2, 6.6, 7.7, 9.9],
        [],
    ]
    content = ak.contents.NumpyArray(
        np.array([3.3, 1.1, 2.2, 0.0, 4.4, 9.9, 6.6, 7.7, 8.8, 5.5])
    )
    offsets = ak.index.Index64(np.array([0, 0, 0, 0, 3, 3, 5, 6, 10, 10, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    assert to_list(listoffsetarray) == [
        [],
        [],
        [],
        [3.3, 1.1, 2.2],
        [],
        [0.0, 4.4],
        [9.9],
        [6.6, 7.7, 8.8, 5.5],
        [],
        [],
        [],
    ]
    assert ak._do.is_unique(listoffsetarray) is True
    assert to_list(ak._do.unique(listoffsetarray, axis=-1)) == [
        [],
        [],
        [],
        [1.1, 2.2, 3.3],
        [],
        [0.0, 4.4],
        [9.9],
        [5.5, 6.6, 7.7, 8.8],
        [],
        [],
        [],
    ]


def test_RegularArray():
    content = ak.contents.NumpyArray(
        np.array(
            [
                2.1,
                8.4,
                7.4,
                1.6,
                2.2,
                3.4,
                6.2,
                5.4,
                1.5,
                3.9,
                3.8,
                3.0,
                8.5,
                6.9,
                4.3,
                3.6,
                6.7,
                1.8,
                3.2,
            ]
        )
    )
    index = ak.index.Index64(
        np.array([13, 9, 13, 4, 8, 3, 15, -1, 16, 2, 8], dtype=np.int64)
    )
    indexedarray = ak.contents.IndexedOptionArray(index, content)
    regular_array = ak.contents.RegularArray(indexedarray, 3, zeros_length=0)

    assert to_list(regular_array) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
    ]
    assert ak._do.is_unique(regular_array) is False
    assert to_list(ak._do.unique(regular_array, axis=-1)) == [
        [3.9, 6.9],
        [1.5, 1.6, 2.2],
        [3.6, 6.7, None],
    ]

    assert to_list(ak._do.unique(regular_array, axis=None)) == [
        1.5,
        1.6,
        2.2,
        3.6,
        3.9,
        6.7,
        6.9,
        None,
    ]

    index2 = ak.index.Index64(
        np.array([13, 9, 13, 9, 13, 13, -1, -1, -1, 2, 8], dtype=np.int64)
    )
    indexedarray2 = ak.contents.IndexedOptionArray(index2, content)
    regular_array2 = ak.contents.RegularArray(indexedarray2, 3, zeros_length=0)
    assert to_list(regular_array2) == [
        [6.9, 3.9, 6.9],
        [3.9, 6.9, 6.9],
        [None, None, None],
    ]
    assert ak._do.is_unique(regular_array2) is False
    assert to_list(ak._do.unique(regular_array2, axis=None)) == [3.9, 6.9, None]


def test_IndexedArray():
    content = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    index = ak.index.Index64(np.array([4, 3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.contents.IndexedArray(index, content)

    assert ak._do.is_unique(indexedarray) is True

    listoffsetarray = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )

    index = ak.index.Index64(np.array([4, 3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.contents.IndexedArray(index, listoffsetarray)
    assert to_list(indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
    ]
    assert ak._do.is_unique(indexedarray) is True

    assert ak._do.is_unique(indexedarray) is True
    assert to_list(ak._do.unique(indexedarray)) == [
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
    assert to_list(ak._do.unique(indexedarray, axis=-1)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
    ]

    index = ak.index.Index64(np.array([4, 4, 3, 2, -1, 0], dtype=np.int64))
    indexedarray = ak.contents.IndexedOptionArray(index, listoffsetarray)
    assert to_list(indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        None,
        [0.0, 1.1, 2.2],
    ]
    assert ak._do.is_unique(indexedarray) is False
    assert ak._do.is_unique(indexedarray, axis=-1) is True
    assert to_list(ak._do.unique(indexedarray, axis=-1)) == [
        [6.6, 7.7, 8.8, 9.9],
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [0.0, 1.1, 2.2],
        None,
    ]


def test_RecordArray():
    array = ak.highlevel.Array(
        [
            {"x": 0.0, "y": []},
            {"x": 8.0, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 1, 3]},
            {"x": 4.4, "y": [4, 1, 1, 4]},
            {"x": 5.5, "y": [5, 4, 5]},
            {"x": 1.1, "y": [6, 1]},
            {"x": 7.7, "y": [7]},
            {"x": 10.0, "y": [99]},
        ]
    )

    assert ak._do.is_unique(array.layout) is False
    assert ak._do.is_unique(array["x"].layout) is True
    assert ak._do.is_unique(array["y"].layout) is False


def test_same_categories():
    categories = ak.highlevel.Array(["one", "two", "three"])
    index1 = ak.index.Index64(np.array([0, 2, 2, 1, 2, 0, 1, 0], dtype=np.int64))
    index2 = ak.index.Index64(np.array([1, 1, 2, 1, 0, 0, 0, 1], dtype=np.int64))
    categorical1 = ak.contents.IndexedArray(
        index1, categories.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak.contents.IndexedArray(
        index2, categories.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak.highlevel.Array(categorical1)
    array2 = ak.highlevel.Array(categorical2)

    assert to_list(ak.sort(categorical1, 0, highlevel=False)) == [
        "one",
        "one",
        "one",
        "three",
        "three",
        "three",
        "two",
        "two",
    ]
    assert ak._do.is_unique(categorical1) is False
    assert ak._do.is_unique(categorical1.content) is True
    assert ak._do.is_unique(categorical2) is False

    assert to_list(array1) == [
        "one",
        "three",
        "three",
        "two",
        "three",
        "one",
        "two",
        "one",
    ]

    assert to_list(array2) == [
        "two",
        "two",
        "three",
        "two",
        "one",
        "one",
        "one",
        "two",
    ]

    assert (array1 == array2).to_list() == [
        False,
        False,
        True,
        True,
        False,
        True,
        False,
        False,
    ]


def test_UnionArray():
    content1 = ak.operations.from_iter([[], [1], [2, 2], [3, 3, 3]], highlevel=False)
    content2 = ak.operations.from_iter(
        [[3.3, 3.3, 3.3], [2.2, 2.2], [1.1], []], highlevel=False
    )
    tags = ak.index.Index8(np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64))
    array = ak.contents.UnionArray.simplified(tags, index, [content1, content2])

    assert to_list(array) == [
        [],
        [3.3, 3.3, 3.3],
        [1],
        [2.2, 2.2],
        [2, 2],
        [1.1],
        [3, 3, 3],
        [],
    ]
    assert ak._do.is_unique(array) is False
    assert to_list(ak._do.unique(array, axis=None)) == [1.0, 1.1, 2.0, 2.2, 3.0, 3.3]
    assert to_list(ak._do.unique(array, axis=-1)) == [
        [],
        [3.3],
        [1.0],
        [2.2],
        [2.0],
        [1.1],
        [3.0],
        [],
    ]
