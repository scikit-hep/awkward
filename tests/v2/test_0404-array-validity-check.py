# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward as ak

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_BitMaskedArray():
    content = ak.layout.NumpyArray(np.arange(13))
    mask = ak.layout.IndexU8(np.array([58, 59], dtype=np.uint8))
    array = ak.layout.BitMaskedArray(
        mask, content, valid_when=False, length=13, lsb_order=False
    )
    array = v1_to_v2(array)
    assert np.asarray(array.mask_as_bool()).tolist() == [
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
    assert np.asarray(array.toByteMaskedArray().mask).tolist() == [
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
    assert np.asarray(array.toIndexedOptionArray64().index).tolist() == [
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
    assert ak.to_list(array) == [
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
    assert array.is_unique(axis=None) is True
    assert ak.to_list(array.unique(axis=None)) == [0, 1, 5, 7, 8, 9, None]
    assert ak.to_list(array.unique(axis=-1)) == [0, 1, 5, 7, 8, 9, None]


def test_ByteMaskedArray():
    content = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    array = v1_to_v2(array)
    assert ak.to_list(array) == [[0.0, 1.1, 2.2], [], None, None, [6.6, 7.7, 8.8, 9.9]]
    assert array.is_unique(axis=None) is True
    assert ak.to_list(array.unique(axis=None)) == [
        0.0,
        1.1,
        2.2,
        6.6,
        7.7,
        8.8,
        9.9,
        None,
    ]
    assert ak.to_list(array.unique(axis=-1)) == [
        [0.0, 1.1, 2.2],
        [],
        [6.6, 7.7, 8.8, 9.9],
        None,
        None,
    ]

    content = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [1.1, 2.2], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    array = v1_to_v2(array)
    assert ak.to_list(array) == [[0.0, 1.1, 2.2], [], None, None, [6.6, 7.7, 8.8, 9.9]]
    assert array.is_unique(axis=None) is True
    assert ak.to_list(array.unique(axis=None)) == [
        0.0,
        1.1,
        2.2,
        6.6,
        7.7,
        8.8,
        9.9,
        None,
    ]
    assert ak.to_list(array.unique(axis=-1)) == [
        [0.0, 1.1, 2.2],
        [],
        [6.6, 7.7, 8.8, 9.9],
        None,
        None,
    ]

    content = ak.from_iter(
        [[1.1, 1.1, 2.2], [], [1.1, 2.2], [5.5], [1.1, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    array = v1_to_v2(array)
    assert ak.to_list(array) == [[1.1, 1.1, 2.2], [], None, None, [1.1, 7.7, 8.8, 9.9]]
    assert array.is_unique(axis=None) is False
    assert ak.to_list(array.unique(axis=None)) == [1.1, 2.2, 7.7, 8.8, 9.9, None]
    assert ak.to_list(array.unique(axis=-1)) == [
        [1.1, 2.2],
        [],
        [1.1, 7.7, 8.8, 9.9],
        None,
        None,
    ]


def test_UnmaskedArray():
    content = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)
    )
    array = ak.layout.UnmaskedArray(content)
    array = v1_to_v2(array)
    assert ak.to_list(array) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert array.is_unique(axis=None) is True
    assert ak.to_list(array.unique(axis=None)) == [1.1, 2.2, 3.3, 4.4, 5.5]


def test_subranges_equal():
    array = ak.layout.NumpyArray(
        np.array(
            [
                [3.3, 2.2, 5.5, 1.1, 4.4],
                [4.4, 2.2, 1.1, 3.3, 5.5],
                [2.2, 1.1, 4.4, 3.3, 5.5],
            ]
        )  # ,
        #     highlevel=False,
    )
    array = v1_to_v2(array)

    starts = ak._v2.index.Index64(np.array([0, 5, 10]))
    stops = ak._v2.index.Index64(np.array([5, 10, 15]))

    result = array.sort(axis=-1).content._subranges_equal(starts, stops, 15)
    assert result is True

    starts = ak._v2.index.Index64(np.array([0, 7]))
    stops = ak._v2.index.Index64(np.array([7, 15]))

    assert array.sort(axis=-1).content._subranges_equal(starts, stops, 15) is False

    starts = ak._v2.index.Index64(np.array([0]))
    stops = ak._v2.index.Index64(np.array([15]))

    assert array.sort(axis=-1).content._subranges_equal(starts, stops, 15) is False

    # FIXME:
    # starts = ak._v2.index.Index64(np.array([0]))
    # stops = ak._v2.index.Index64(np.array([5]))
    #
    # assert array.sort(axis=-1).content._subranges_equal(starts, stops, 15) is True


def test_categorical():
    array = ak.Array(["1chchc", "1chchc", "2sss", "3", "4", "5"])
    categorical = ak.to_categorical(array)
    assert ak.is_valid(categorical) is True
    assert categorical.layout.is_unique() is False

    array = v1_to_v2(array.layout)
    categorical = v1_to_v2(categorical.layout)

    assert ak.is_valid(ak.Array(array)) is True
    assert array.is_unique() is False
    assert ak.is_valid(ak.Array(categorical)) is True
    assert categorical.is_unique() is True


def test_NumpyArray():
    array = ak.layout.NumpyArray(np.array([5, 6, 1, 3, 4, 5]))
    array = v1_to_v2(array)
    assert array.is_unique() is False
    assert ak.to_list(array.unique()) == [1, 3, 4, 5, 6]


def test_2d():
    nparray = np.array(
        [
            [3.3, 2.2, 5.5, 1.1, 4.4],
            [4.4, 2.2, 1.1, 3.3, 5.5],
            [2.2, 1.1, 4.4, 3.3, 5.5],
        ]
    )

    assert ak.to_list(np.unique(nparray)) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert ak.to_list(np.unique(nparray, axis=0)) == [
        [2.2, 1.1, 4.4, 3.3, 5.5],
        [3.3, 2.2, 5.5, 1.1, 4.4],
        [4.4, 2.2, 1.1, 3.3, 5.5],
    ]
    assert ak.to_list(np.unique(nparray, axis=1)) == [
        [1.1, 2.2, 3.3, 4.4, 5.5],
        [3.3, 2.2, 4.4, 5.5, 1.1],
        [3.3, 1.1, 2.2, 5.5, 4.4],
    ]
    assert ak.to_list(np.unique(nparray, axis=-1)) == [
        [1.1, 2.2, 3.3, 4.4, 5.5],
        [3.3, 2.2, 4.4, 5.5, 1.1],
        [3.3, 1.1, 2.2, 5.5, 4.4],
    ]
    assert ak.to_list(np.unique(nparray, axis=-2)) == [
        [2.2, 1.1, 4.4, 3.3, 5.5],
        [3.3, 2.2, 5.5, 1.1, 4.4],
        [4.4, 2.2, 1.1, 3.3, 5.5],
    ]

    array = ak.layout.NumpyArray(
        np.array(
            [
                [3.3, 2.2, 5.5, 1.1, 4.4],
                [4.4, 2.2, 1.1, 3.3, 5.5],
                [2.2, 1.1, 4.4, 3.3, 5.5],
            ]
        )
    )
    array = v1_to_v2(array)

    assert array.is_unique() is False
    assert array.is_unique(axis=-1) is True

    assert ak.to_list(array.unique()) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert ak.to_list(array.unique(axis=-1)) == [
        [1.1, 2.2, 3.3, 4.4, 5.5],
        [1.1, 2.2, 3.3, 4.4, 5.5],
        [1.1, 2.2, 3.3, 4.4, 5.5],
    ]


def test_2d_in_axis():
    array = ak.layout.NumpyArray(
        np.array(
            [
                [3.3, 2.2, 5.5, 1.1, 4.4, 1.1],
                [4.4, 2.2, 1.1, 3.3, 5.5, 3.3],
                [2.2, 2.2, 1.1, 4.4, 3.3, 5.5],
            ]
        )
    )
    assert ak.to_list(ak.sort(array, axis=-1)) == [
        [1.1, 1.1, 2.2, 3.3, 4.4, 5.5],
        [1.1, 2.2, 3.3, 3.3, 4.4, 5.5],
        [1.1, 2.2, 2.2, 3.3, 4.4, 5.5],
    ]
    array = v1_to_v2(array)

    assert array.is_unique() is False
    assert ak.to_list(array.unique()) == [1.1, 2.2, 3.3, 4.4, 5.5]

    assert ak.to_list(array.sort(axis=-1)) == [
        [1.1, 1.1, 2.2, 3.3, 4.4, 5.5],
        [1.1, 2.2, 3.3, 3.3, 4.4, 5.5],
        [1.1, 2.2, 2.2, 3.3, 4.4, 5.5],
    ]
    assert ak.to_list(array.unique(axis=-1)) == [
        [1.1, 2.2, 3.3, 4.4, 5.5],
        [1.1, 2.2, 3.3, 4.4, 5.5],
        [1.1, 2.2, 3.3, 4.4, 5.5],
    ]


def test_3d():
    array = ak.layout.NumpyArray(
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
    array = v1_to_v2(array)
    assert array.is_unique() is True

    assert ak.to_list(array.unique()) == [
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

    assert ak.to_list(array.unique(axis=-1)) == [
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

    array = ak.layout.NumpyArray(np_array)
    array = v1_to_v2(array)

    assert ak.to_list(array.unique()) == ak.to_list(np.unique(np_array))
    assert ak.to_list(array.unique(axis=-1)) == [
        [[1.1, 4.4, 5.5, 13.13], [7.7, 10.1, 13.13], [12.12, 13.13]],
        [
            [-5.5, -4.4, -2.2, -1.1, 13.13],
            [-10.1, -9.9, -7.7, 13.13],
            [-15.15, -13.13, -12.12, 13.13],
        ],
    ]


def test_ListOffsetArray():
    array = ak.from_iter(["one", "two", "three", "four", "five"], highlevel=False)
    array = v1_to_v2(array)
    assert ak.to_list(array.sort(0, True, True)) == [
        "five",
        "four",
        "one",
        "three",
        "two",
    ]
    assert array.is_unique() is True
    assert ak.to_list(array.unique(axis=-1)) == [
        "five",
        "four",
        "one",
        "three",
        "two",
    ]

    array2 = ak.from_iter(["one", "two", "one", "four", "two"], highlevel=False)
    array2 = v1_to_v2(array2)
    assert ak.to_list(array2.sort(0, True, True)) == [
        "four",
        "one",
        "one",
        "two",
        "two",
    ]
    assert array2.is_unique() is False
    assert ak.to_list(array2.unique(axis=-1)) == [
        "four",
        "one",
        "two",
    ]

    content = ak.layout.NumpyArray(
        np.array([3.3, 1.1, 2.2, 0.0, 4.4, 9.9, 6.6, 7.7, 8.8, 5.5])
    )
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)
    listoffsetarray = v1_to_v2(listoffsetarray)
    assert ak.to_list(listoffsetarray) == [
        [3.3, 1.1, 2.2],
        [],
        [0.0, 4.4],
        [9.9],
        [6.6, 7.7, 8.8, 5.5],
        [],
    ]
    assert listoffsetarray.is_unique() is True
    assert ak.to_list(listoffsetarray.unique(axis=-1)) == [
        [1.1, 2.2, 3.3],
        [],
        [0.0, 4.4],
        [9.9],
        [5.5, 6.6, 7.7, 8.8],
        [],
    ]
    assert ak.to_list(listoffsetarray.unique()) == [
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

    content = ak.layout.NumpyArray(
        np.array([3.3, 1.1, 2.2, 0.0, 4.4, 9.9, 2.2, 3.3, 1.1, 5.5])
    )
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 9, 10]))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)
    listoffsetarray = v1_to_v2(listoffsetarray)
    assert ak.to_list(listoffsetarray) == [
        [3.3, 1.1, 2.2],
        [],
        [0.0, 4.4],
        [9.9],
        [2.2, 3.3, 1.1],
        [5.5],
    ]
    assert listoffsetarray.is_unique() is False
    assert ak.to_list(listoffsetarray.unique(axis=-1)) == [
        [1.1, 2.2, 3.3],
        [],
        [0.0, 4.4],
        [9.9],
        [1.1, 2.2, 3.3],
        [5.5],
    ]

    content1 = ak.layout.NumpyArray(
        np.array([3.3, 1.1, 2.2, 2.2, 3.3, 1.1, 0.0, 4.4, 9.9, 5.5])
    )
    offsets1 = ak.layout.Index64(np.array([0, 3, 6, 6, 8, 9, 10]))
    listoffsetarray1 = ak.layout.ListOffsetArray64(offsets1, content1)
    listoffsetarray1 = v1_to_v2(listoffsetarray1)
    assert ak.to_list(listoffsetarray1) == [
        [3.3, 1.1, 2.2],
        [2.2, 3.3, 1.1],
        [],
        [0.0, 4.4],
        [9.9],
        [5.5],
    ]
    assert listoffsetarray1.is_unique() is False
    assert ak.to_list(listoffsetarray1.unique(axis=-1)) == [
        [1.1, 2.2, 3.3],
        [1.1, 2.2, 3.3],
        [],
        [0.0, 4.4],
        [9.9],
        [5.5],
    ]

    content2 = ak.layout.NumpyArray(
        np.array([3.3, 1.1, 2.2, 2.2, 3.3, 1.1, 0.0, 4.4, 9.9, 5.5, 5.5])
    )
    offsets2 = ak.layout.Index64(np.array([0, 3, 6, 6, 8, 9, 10, 11]))
    listoffsetarray2 = ak.layout.ListOffsetArray64(offsets2, content2)
    listoffsetarray2 = v1_to_v2(listoffsetarray2)
    assert ak.to_list(listoffsetarray2) == [
        [3.3, 1.1, 2.2],
        [2.2, 3.3, 1.1],
        [],
        [0.0, 4.4],
        [9.9],
        [5.5],
        [5.5],
    ]
    assert listoffsetarray2.is_unique() is False
    assert ak.to_list(listoffsetarray2.unique(axis=-1)) == [
        [1.1, 2.2, 3.3],
        [1.1, 2.2, 3.3],
        [],
        [0.0, 4.4],
        [9.9],
        [5.5],
        [5.5],
    ]

    content2 = ak.layout.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 1.1, 5.5, 6.6, 7.7, 2.2, 9.9])
    )
    offsets2 = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray2 = ak.layout.ListOffsetArray64(offsets2, content2)
    listoffsetarray2 = v1_to_v2(listoffsetarray2)
    assert ak.to_list(listoffsetarray2) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 1.1],
        [5.5],
        [6.6, 7.7, 2.2, 9.9],
        [],
    ]
    assert ak.to_list(listoffsetarray2.sort()) == [
        [0.0, 1.1, 2.2],
        [],
        [1.1, 3.3],
        [5.5],
        [2.2, 6.6, 7.7, 9.9],
        [],
    ]
    assert listoffsetarray2.is_unique(axis=-1) is True
    assert ak.to_list(listoffsetarray2.unique(axis=-1)) == [
        [0.0, 1.1, 2.2],
        [],
        [1.1, 3.3],
        [5.5],
        [2.2, 6.6, 7.7, 9.9],
        [],
    ]
    content = ak.layout.NumpyArray(
        np.array([3.3, 1.1, 2.2, 0.0, 4.4, 9.9, 6.6, 7.7, 8.8, 5.5])
    )
    offsets = ak.layout.Index64(np.array([0, 0, 0, 0, 3, 3, 5, 6, 10, 10, 10, 10]))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)
    listoffsetarray = v1_to_v2(listoffsetarray)
    assert ak.to_list(listoffsetarray) == [
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
    assert listoffsetarray.is_unique() is True
    assert ak.to_list(listoffsetarray.unique(axis=-1)) == [
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
    content = ak.layout.NumpyArray(
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
    index = ak.layout.Index64(
        np.array([13, 9, 13, 4, 8, 3, 15, -1, 16, 2, 8], dtype=np.int64)
    )
    indexedarray = ak.layout.IndexedOptionArray64(index, content)
    regular_array = ak.layout.RegularArray(indexedarray, 3, zeros_length=0)
    regular_array = v1_to_v2(regular_array)
    assert ak.to_list(regular_array) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
    ]
    assert regular_array.is_unique() is False
    assert ak.to_list(regular_array.unique(axis=-1)) == [
        [3.9, 6.9],
        [1.5, 1.6, 2.2],
        [3.6, 6.7, None],
    ]

    assert ak.to_list(regular_array.unique(axis=None)) == [
        1.5,
        1.6,
        2.2,
        3.6,
        3.9,
        6.7,
        6.9,
        None,
    ]

    index2 = ak.layout.Index64(
        np.array([13, 9, 13, 9, 13, 13, -1, -1, -1, 2, 8], dtype=np.int64)
    )
    indexedarray2 = ak.layout.IndexedOptionArray64(index2, content)
    regular_array2 = ak.layout.RegularArray(indexedarray2, 3, zeros_length=0)
    regular_array2 = v1_to_v2(regular_array2)
    assert ak.to_list(regular_array2) == [
        [6.9, 3.9, 6.9],
        [3.9, 6.9, 6.9],
        [None, None, None],
    ]
    assert regular_array2.is_unique() is False
    assert ak.to_list(regular_array2.unique(axis=None)) == [3.9, 6.9, None]


def test_IndexedArray():
    listoffsetarray = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )

    index = ak.layout.Index64(np.array([4, 3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, listoffsetarray)
    assert ak.to_list(indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
    ]
    assert indexedarray.is_unique() is True
    indexedarray = v1_to_v2(indexedarray)
    assert indexedarray.is_unique() is True
    assert ak.to_list(indexedarray.unique()) == [
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
    assert ak.to_list(indexedarray.unique(axis=-1)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
    ]

    index = ak.layout.Index64(np.array([4, 4, 3, 2, -1, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedOptionArray64(index, listoffsetarray)
    assert ak.to_list(indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        None,
        [0.0, 1.1, 2.2],
    ]
    assert indexedarray.is_unique() is False
    indexedarray = v1_to_v2(indexedarray)
    assert indexedarray.is_unique() is False
    assert indexedarray.is_unique(axis=-1) is True
    assert ak.to_list(indexedarray.unique(axis=-1)) == [
        [6.6, 7.7, 8.8, 9.9],
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [0.0, 1.1, 2.2],
        None,
    ]


def test_RecordArray():
    array = ak.Array(
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
    array = v1_to_v2(array.layout)

    assert array.is_unique() is False
    assert array["x"].is_unique() is True
    assert array["y"].is_unique() is False


def test_same_categories():
    categories = ak.Array(["one", "two", "three"])
    index1 = ak.layout.Index64(np.array([0, 2, 2, 1, 2, 0, 1, 0], dtype=np.int64))
    index2 = ak.layout.Index64(np.array([1, 1, 2, 1, 0, 0, 0, 1], dtype=np.int64))
    categorical1 = ak.layout.IndexedArray64(
        index1, categories.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak.layout.IndexedArray64(
        index2, categories.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak.Array(categorical1)
    array2 = ak.Array(categorical2)

    categorical1 = v1_to_v2(categorical1)
    categorical2 = v1_to_v2(categorical2)

    assert ak.to_list(categorical1.sort(0, True, True)) == [
        "one",
        "one",
        "one",
        "three",
        "three",
        "three",
        "two",
        "two",
    ]
    assert categorical1.is_unique() is False
    assert categorical1.content.is_unique() is True
    assert categorical2.is_unique() is False

    array1 = v1_to_v2(array1.layout)
    assert ak.to_list(array1) == [
        "one",
        "three",
        "three",
        "two",
        "three",
        "one",
        "two",
        "one",
    ]

    array2 = v1_to_v2(array2.layout)

    assert ak.to_list(array2) == [
        "two",
        "two",
        "three",
        "two",
        "one",
        "one",
        "one",
        "two",
    ]

    assert (array1 == array2) is False


def test_UnionArray():
    content1 = ak.from_iter([[], [1], [2, 2], [3, 3, 3]], highlevel=False)
    content2 = ak.from_iter([[3.3, 3.3, 3.3], [2.2, 2.2], [1.1], []], highlevel=False)
    tags = ak.layout.Index8(np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8))
    index = ak.layout.Index64(np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64))
    array = ak.layout.UnionArray8_64(tags, index, [content1, content2])
    array = v1_to_v2(array)
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
    assert array.is_unique() is False
    assert ak.to_list(array.unique(axis=None)) == [1.0, 1.1, 2.0, 2.2, 3.0, 3.3]
    assert ak.to_list(array.unique(axis=-1)) == [
        [],
        [3.3],
        [1.0],
        [2.2],
        [2.0],
        [1.1],
        [3.0],
        [],
    ]
