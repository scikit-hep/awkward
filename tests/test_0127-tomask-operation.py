# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_ByteMaskedArray():
    content = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    assert ak.to_list(array) == [[0.0, 1.1, 2.2], [], None, None, [6.6, 7.7, 8.8, 9.9]]
    assert ak.to_list(array[-1]) == [6.6, 7.7, 8.8, 9.9]
    assert ak.to_list(array[-2]) == None
    assert ak.to_list(array[1:]) == [[], None, None, [6.6, 7.7, 8.8, 9.9]]
    assert array[4, 1] == 7.7
    assert ak.to_list(array[2:, 1]) == [None, None, 7.7]
    assert ak.to_list(array[2:, [2, 1, 1, 0]]) == [None, None, [8.8, 7.7, 7.7, 6.6]]

    content = ak.from_iter(
        [
            {"x": 0.0, "y": []},
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 3, 3]},
            {"x": 4.4, "y": [4, 4, 4, 4]},
        ],
        highlevel=False,
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    assert ak.to_list(array) == [
        {"x": 0.0, "y": []},
        {"x": 1.1, "y": [1]},
        None,
        None,
        {"x": 4.4, "y": [4, 4, 4, 4]},
    ]
    assert ak.to_list(array["x"]) == [0.0, 1.1, None, None, 4.4]
    assert ak.to_list(array[["x", "y"]]) == [
        {"x": 0.0, "y": []},
        {"x": 1.1, "y": [1]},
        None,
        None,
        {"x": 4.4, "y": [4, 4, 4, 4]},
    ]


def test_ByteMaskedArray_jaggedslice0():
    array = ak.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    index = ak.layout.Index64(np.array([0, 1, 2, 3], dtype=np.int64))
    indexedarray = ak.layout.IndexedOptionArray64(index, array)
    assert ak.to_list(indexedarray) == [
        [0.0, 1.1, 2.2],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(indexedarray[ak.Array([[0, -1], [0], [], [1, 1]])]) == [
        [0.0, 2.2],
        [3.3],
        [],
        [7.7, 7.7],
    ]

    mask = ak.layout.Index8(np.array([0, 0, 0, 0], dtype=np.int8))
    maskedarray = ak.layout.ByteMaskedArray(mask, array, valid_when=False)
    assert ak.to_list(maskedarray) == [
        [0.0, 1.1, 2.2],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(maskedarray[ak.Array([[0, -1], [0], [], [1, 1]])]) == [
        [0.0, 2.2],
        [3.3],
        [],
        [7.7, 7.7],
    ]


def test_ByteMaskedArray_jaggedslice1():
    model = ak.Array(
        [[0.0, 1.1, None, 2.2], [], [3.3, None, 4.4], [5.5], [6.6, 7.7, None, 8.8, 9.9]]
    )
    assert ak.to_list(model[ak.Array([[3, 2, 1, 1, 0], [], [1], [0, 0], [1, 2]])]) == [
        [2.2, None, 1.1, 1.1, 0.0],
        [],
        [None],
        [5.5, 5.5],
        [7.7, None],
    ]

    content = ak.layout.NumpyArray(
        np.array([0.0, 1.1, 999, 2.2, 3.3, 123, 4.4, 5.5, 6.6, 7.7, 321, 8.8, 9.9])
    )
    mask = ak.layout.Index8(
        np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.int8)
    )
    maskedarray = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    offsets = ak.layout.Index64(np.array([0, 4, 4, 7, 8, 13], dtype=np.int64))
    listarray = ak.Array(ak.layout.ListOffsetArray64(offsets, maskedarray))
    assert ak.to_list(listarray) == ak.to_list(model)
    assert ak.to_list(
        listarray[ak.Array([[3, 2, 1, 1, 0], [], [1], [0, 0], [1, 2]])]
    ) == [[2.2, None, 1.1, 1.1, 0.0], [], [None], [5.5, 5.5], [7.7, None]]


def test_ByteMaskedArray_jaggedslice2():
    model = ak.Array(
        [
            [[0.0, 1.1, None, 2.2], [], [3.3, None, 4.4]],
            [],
            [[5.5]],
            [[6.6, 7.7, None, 8.8, 9.9]],
        ]
    )
    assert ak.to_list(
        model[ak.Array([[[3, 2, 1, 1, 0], [], [1]], [], [[0, 0]], [[1, 2]]])]
    ) == [[[2.2, None, 1.1, 1.1, 0.0], [], [None]], [], [[5.5, 5.5]], [[7.7, None]]]

    content = ak.layout.NumpyArray(
        np.array([0.0, 1.1, 999, 2.2, 3.3, 123, 4.4, 5.5, 6.6, 7.7, 321, 8.8, 9.9])
    )
    mask = ak.layout.Index8(
        np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.int8)
    )
    maskedarray = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    offsets = ak.layout.Index64(np.array([0, 4, 4, 7, 8, 13], dtype=np.int64))
    sublistarray = ak.layout.ListOffsetArray64(offsets, maskedarray)
    offsets2 = ak.layout.Index64(np.array([0, 3, 3, 4, 5], dtype=np.int64))
    listarray = ak.Array(ak.layout.ListOffsetArray64(offsets2, sublistarray))
    assert ak.to_list(listarray) == ak.to_list(model)
    assert ak.to_list(
        listarray[ak.Array([[[3, 2, 1, 1, 0], [], [1]], [], [[0, 0]], [[1, 2]]])]
    ) == [[[2.2, None, 1.1, 1.1, 0.0], [], [None]], [], [[5.5, 5.5]], [[7.7, None]]]


def test_ByteMaskedArray_jaggedslice3():
    model = ak.Array(
        [
            [[[0.0, 1.1, None, 2.2], [], [3.3, None, 4.4]], []],
            [[[5.5]], [[6.6, 7.7, None, 8.8, 9.9]]],
        ]
    )
    assert ak.to_list(
        model[ak.Array([[[[3, 2, 1, 1, 0], [], [1]], []], [[[0, 0]], [[1, 2]]]])]
    ) == [[[[2.2, None, 1.1, 1.1, 0.0], [], [None]], []], [[[5.5, 5.5]], [[7.7, None]]]]

    content = ak.layout.NumpyArray(
        np.array([0.0, 1.1, 999, 2.2, 3.3, 123, 4.4, 5.5, 6.6, 7.7, 321, 8.8, 9.9])
    )
    mask = ak.layout.Index8(
        np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.int8)
    )
    maskedarray = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    offsets = ak.layout.Index64(np.array([0, 4, 4, 7, 8, 13], dtype=np.int64))
    subsublistarray = ak.layout.ListOffsetArray64(offsets, maskedarray)
    offsets2 = ak.layout.Index64(np.array([0, 3, 3, 4, 5], dtype=np.int64))
    sublistarray = ak.layout.ListOffsetArray64(offsets2, subsublistarray)
    offsets3 = ak.layout.Index64(np.array([0, 2, 4], dtype=np.int64))
    listarray = ak.Array(ak.layout.ListOffsetArray64(offsets3, sublistarray))
    assert ak.to_list(listarray) == ak.to_list(model)
    assert ak.to_list(
        listarray[ak.Array([[[[3, 2, 1, 1, 0], [], [1]], []], [[[0, 0]], [[1, 2]]]])]
    ) == [[[[2.2, None, 1.1, 1.1, 0.0], [], [None]], []], [[[5.5, 5.5]], [[7.7, None]]]]


def test_ByteMaskedArray_to_slice():
    content = ak.layout.NumpyArray(np.array([5, 2, 999, 3, 9, 123, 1], dtype=np.int64))
    mask = ak.layout.Index8(np.array([0, 0, 1, 0, 0, 1, 0], dtype=np.int8))
    maskedarray = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    assert ak.to_list(maskedarray) == [5, 2, None, 3, 9, None, 1]

    assert (
        ak._ext._slice_tostring(maskedarray)
        == "[missing([0, 1, -1, ..., 3, -1, 4], array([5, 2, 3, 9, 1]))]"
    )


def test_ByteMaskedArray_as_slice():
    array = ak.Array(
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], check_valid=True
    )
    slicecontent = ak.from_iter([3, 6, 999, 123, -2, 6], highlevel=False)
    slicemask = ak.layout.Index8(np.array([0, 0, 1, 1, 0, 0], dtype=np.int8))
    slicearray = ak.layout.ByteMaskedArray(slicemask, slicecontent, valid_when=False)

    assert ak.to_list(array[slicearray]) == [3.3, 6.6, None, None, 8.8, 6.6]


def test_ByteMaskedArray_setidentities():
    content = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9], [321]],
        highlevel=False,
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    assert ak.to_list(array) == [[0.0, 1.1, 2.2], [], None, None, [6.6, 7.7, 8.8, 9.9]]
    array.setidentities()
    assert np.asarray(array.identities).tolist() == [[0], [1], [2], [3], [4]]
    assert np.asarray(array.content.identities).tolist() == [
        [0],
        [1],
        [2],
        [3],
        [4],
        [-1],
    ]

    assert ak.is_valid(ak.Array(array))


def test_ByteMaskedArray_num():
    content = ak.from_iter(
        [
            [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 7.7, 8.8, 9.9]],
            [[], [10.0, 11.1, 12.2]],
        ],
        highlevel=False,
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.Array(ak.layout.ByteMaskedArray(mask, content, valid_when=False))
    assert ak.to_list(array) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert ak.num(array, axis=0) == 5
    assert ak.num(array, axis=-3) == 5
    assert ak.to_list(ak.num(array, axis=1)) == [3, 0, None, None, 2]
    assert ak.to_list(ak.num(array, axis=-2)) == [3, 0, None, None, 2]
    assert ak.to_list(ak.num(array, axis=2)) == [[3, 0, 2], [], None, None, [0, 3]]
    assert ak.to_list(ak.num(array, axis=-1)) == [[3, 0, 2], [], None, None, [0, 3]]


def test_ByteMaskedArray_flatten():
    content = ak.from_iter(
        [
            [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 7.7, 8.8, 9.9]],
            [[], [10.0, 11.1, 12.2]],
        ],
        highlevel=False,
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.Array(ak.layout.ByteMaskedArray(mask, content, valid_when=False))
    assert ak.to_list(array) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert ak.to_list(ak.flatten(array, axis=1)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [],
        [10.0, 11.1, 12.2],
    ]
    assert ak.to_list(ak.flatten(array, axis=-2)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [],
        [10.0, 11.1, 12.2],
    ]
    assert ak.to_list(ak.flatten(array, axis=2)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        None,
        None,
        [10.0, 11.1, 12.2],
    ]
    assert ak.to_list(ak.flatten(array, axis=-1)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        None,
        None,
        [10.0, 11.1, 12.2],
    ]


def test_IndexedOptionArray_pad_none():
    array = ak.Array(
        [[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], None, None, [[], [10.0, 11.1, 12.2]]]
    )
    assert ak.to_list(array) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert ak.to_list(ak.pad_none(array, 7, axis=0)) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
        None,
        None,
    ]
    assert ak.to_list(ak.pad_none(array, 7, axis=-3)) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
        None,
        None,
    ]
    assert ak.to_list(ak.pad_none(array, 3, axis=1)) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [None, None, None],
        None,
        None,
        [[], [10.0, 11.1, 12.2], None],
    ]
    assert ak.to_list(ak.pad_none(array, 3, axis=-2)) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [None, None, None],
        None,
        None,
        [[], [10.0, 11.1, 12.2], None],
    ]
    assert ak.to_list(ak.pad_none(array, 3, axis=2)) == [
        [[0.0, 1.1, 2.2], [None, None, None], [3.3, 4.4, None]],
        [],
        None,
        None,
        [[None, None, None], [10.0, 11.1, 12.2]],
    ]
    assert ak.to_list(ak.pad_none(array, 3, axis=-1)) == [
        [[0.0, 1.1, 2.2], [None, None, None], [3.3, 4.4, None]],
        [],
        None,
        None,
        [[None, None, None], [10.0, 11.1, 12.2]],
    ]
    assert ak.to_list(ak.pad_none(array, 3, axis=0, clip=True)) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
    ]
    assert ak.to_list(ak.pad_none(array, 3, axis=-3, clip=True)) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
    ]
    assert ak.to_list(ak.pad_none(array, 2, axis=1, clip=True)) == [
        [[0.0, 1.1, 2.2], []],
        [None, None],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert ak.to_list(ak.pad_none(array, 2, axis=-2, clip=True)) == [
        [[0.0, 1.1, 2.2], []],
        [None, None],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert ak.to_list(ak.pad_none(array, 2, axis=2, clip=True)) == [
        [[0.0, 1.1], [None, None], [3.3, 4.4]],
        [],
        None,
        None,
        [[None, None], [10.0, 11.1]],
    ]
    assert ak.to_list(ak.pad_none(array, 2, axis=-1, clip=True)) == [
        [[0.0, 1.1], [None, None], [3.3, 4.4]],
        [],
        None,
        None,
        [[None, None], [10.0, 11.1]],
    ]


def test_ByteMaskedArray_pad_none():
    content = ak.from_iter(
        [
            [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 7.7, 8.8, 9.9]],
            [[], [10.0, 11.1, 12.2]],
        ],
        highlevel=False,
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.Array(ak.layout.ByteMaskedArray(mask, content, valid_when=False))
    assert ak.to_list(array) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert ak.to_list(ak.pad_none(array, 7, axis=0)) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
        None,
        None,
    ]
    assert ak.to_list(ak.pad_none(array, 7, axis=-3)) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
        None,
        None,
    ]
    assert ak.to_list(ak.pad_none(array, 3, axis=1)) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [None, None, None],
        None,
        None,
        [[], [10.0, 11.1, 12.2], None],
    ]
    assert ak.to_list(ak.pad_none(array, 3, axis=-2)) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [None, None, None],
        None,
        None,
        [[], [10.0, 11.1, 12.2], None],
    ]
    assert ak.to_list(ak.pad_none(array, 3, axis=2)) == [
        [[0.0, 1.1, 2.2], [None, None, None], [3.3, 4.4, None]],
        [],
        None,
        None,
        [[None, None, None], [10.0, 11.1, 12.2]],
    ]
    assert ak.to_list(ak.pad_none(array, 3, axis=-1)) == [
        [[0.0, 1.1, 2.2], [None, None, None], [3.3, 4.4, None]],
        [],
        None,
        None,
        [[None, None, None], [10.0, 11.1, 12.2]],
    ]
    assert ak.to_list(ak.pad_none(array, 3, axis=0, clip=True)) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
    ]
    assert ak.to_list(ak.pad_none(array, 3, axis=-3, clip=True)) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
    ]
    assert ak.to_list(ak.pad_none(array, 2, axis=1, clip=True)) == [
        [[0.0, 1.1, 2.2], []],
        [None, None],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert ak.to_list(ak.pad_none(array, 2, axis=-2, clip=True)) == [
        [[0.0, 1.1, 2.2], []],
        [None, None],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert ak.to_list(ak.pad_none(array, 2, axis=2, clip=True)) == [
        [[0.0, 1.1], [None, None], [3.3, 4.4]],
        [],
        None,
        None,
        [[None, None], [10.0, 11.1]],
    ]
    assert ak.to_list(ak.pad_none(array, 2, axis=-1, clip=True)) == [
        [[0.0, 1.1], [None, None], [3.3, 4.4]],
        [],
        None,
        None,
        [[None, None], [10.0, 11.1]],
    ]


def test_ByteMaskedArray_reduce():
    content = ak.layout.NumpyArray(
        np.array(
            [
                2,
                3,
                5,
                7,
                11,
                0,
                0,
                0,
                0,
                0,
                31,
                37,
                41,
                43,
                47,
                53,
                59,
                61,
                67,
                71,
                0,
                0,
                0,
                0,
                0,
                101,
                103,
                107,
                109,
                113,
            ],
            dtype=np.int64,
        )
    )
    offsets1 = ak.layout.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets1, content)
    # index = ak.layout.Index64(np.array([0, -1, 2, 3, -1, 5], dtype=np.int64))
    # optionarray = ak.layout.IndexedOptionArray64(index, listoffsetarray)
    mask = ak.layout.Index8(np.array([0, 1, 0, 0, 1, 0], dtype=np.int8))
    optionarray = ak.layout.ByteMaskedArray(mask, listoffsetarray, valid_when=False)
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(offsets2, optionarray)
    assert ak.to_list(depth2) == [
        [[2, 3, 5, 7, 11], None, [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], None, [101, 103, 107, 109, 113]],
    ]

    assert ak.to_list(depth2.prod(-1)) == [
        [2 * 3 * 5 * 7 * 11, None, 31 * 37 * 41 * 43 * 47],
        [53 * 59 * 61 * 67 * 71, None, 101 * 103 * 107 * 109 * 113],
    ]

    assert ak.to_list(depth2.prod(-2)) == [
        [31 * 2, 37 * 3, 41 * 5, 43 * 7, 47 * 11],
        [101 * 53, 103 * 59, 107 * 61, 109 * 67, 113 * 71],
    ]

    assert ak.to_list(depth2.prod(-3)) == [
        [53 * 2, 59 * 3, 61 * 5, 67 * 7, 71 * 11],
        [],
        [101 * 31, 103 * 37, 107 * 41, 109 * 43, 113 * 47],
    ]

    content = ak.layout.NumpyArray(
        np.array(
            [
                2,
                3,
                5,
                7,
                11,
                0,
                0,
                0,
                0,
                0,
                31,
                37,
                41,
                43,
                47,
                53,
                59,
                61,
                67,
                71,
                0,
                0,
                0,
                0,
                0,
                101,
                103,
                107,
                109,
                113,
            ],
            dtype=np.int64,
        )
    )
    # index = ak.layout.Index64(np.array([0, 1, 2, 3, 4, -1, -1, -1, -1, -1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, -1, -1, -1, -1, -1, 25, 26, 27, 28, 29], dtype=np.int64))
    # optionarray = ak.layout.IndexedOptionArray64(index, content)
    mask = ak.layout.Index8(
        np.array(
            [
                0,
                0,
                0,
                0,
                0,
                1,
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
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=np.int8,
        )
    )
    optionarray = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    offsets1 = ak.layout.Index64(np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets1, optionarray)
    offsets2 = ak.layout.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak.layout.ListOffsetArray64(offsets2, listoffsetarray)
    assert ak.to_list(depth2) == [
        [[2, 3, 5, 7, 11], [None, None, None, None, None], [31, 37, 41, 43, 47]],
        [
            [53, 59, 61, 67, 71],
            [None, None, None, None, None],
            [101, 103, 107, 109, 113],
        ],
    ]

    assert ak.to_list(depth2.prod(-1)) == [
        [2 * 3 * 5 * 7 * 11, 1 * 1 * 1 * 1 * 1, 31 * 37 * 41 * 43 * 47],
        [53 * 59 * 61 * 67 * 71, 1 * 1 * 1 * 1 * 1, 101 * 103 * 107 * 109 * 113],
    ]

    assert ak.to_list(depth2.prod(-2)) == [
        [31 * 2, 37 * 3, 41 * 5, 43 * 7, 47 * 11],
        [101 * 53, 103 * 59, 107 * 61, 109 * 67, 113 * 71],
    ]

    assert ak.to_list(depth2.prod(-3)) == [
        [53 * 2, 59 * 3, 61 * 5, 67 * 7, 71 * 11],
        [1, 1, 1, 1, 1],
        [101 * 31, 103 * 37, 107 * 41, 109 * 43, 113 * 47],
    ]


def test_ByteMaskedArray_localindex():
    content = ak.from_iter(
        [
            [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 7.7, 8.8, 9.9]],
            [[], [10.0, 11.1, 12.2]],
        ],
        highlevel=False,
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    assert ak.to_list(array) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert ak.to_list(array.localindex(axis=0)) == [0, 1, 2, 3, 4]
    assert ak.to_list(array.localindex(axis=-3)) == [0, 1, 2, 3, 4]
    assert ak.to_list(array.localindex(axis=1)) == [[0, 1, 2], [], None, None, [0, 1]]
    assert ak.to_list(array.localindex(axis=-2)) == [[0, 1, 2], [], None, None, [0, 1]]
    assert ak.to_list(array.localindex(axis=2)) == [
        [[0, 1, 2], [], [0, 1]],
        [],
        None,
        None,
        [[], [0, 1, 2]],
    ]
    assert ak.to_list(array.localindex(axis=-1)) == [
        [[0, 1, 2], [], [0, 1]],
        [],
        None,
        None,
        [[], [0, 1, 2]],
    ]


def test_ByteMaskedArray_combinations():
    content = ak.from_iter(
        [[[0, 1, 2], [], [3, 4]], [], [[5]], [[6, 7, 8, 9]], [[], [10, 11, 12]]],
        highlevel=False,
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.Array(ak.layout.ByteMaskedArray(mask, content, valid_when=False))
    assert ak.to_list(array) == [
        [[0, 1, 2], [], [3, 4]],
        [],
        None,
        None,
        [[], [10, 11, 12]],
    ]
    assert ak.to_list(ak.combinations(array, 2, axis=0)) == [
        ([[0, 1, 2], [], [3, 4]], []),
        ([[0, 1, 2], [], [3, 4]], None),
        ([[0, 1, 2], [], [3, 4]], None),
        ([[0, 1, 2], [], [3, 4]], [[], [10, 11, 12]]),
        ([], None),
        ([], None),
        ([], [[], [10, 11, 12]]),
        (None, None),
        (None, [[], [10, 11, 12]]),
        (None, [[], [10, 11, 12]]),
    ]
    assert ak.to_list(ak.combinations(array, 2, axis=-3)) == [
        ([[0, 1, 2], [], [3, 4]], []),
        ([[0, 1, 2], [], [3, 4]], None),
        ([[0, 1, 2], [], [3, 4]], None),
        ([[0, 1, 2], [], [3, 4]], [[], [10, 11, 12]]),
        ([], None),
        ([], None),
        ([], [[], [10, 11, 12]]),
        (None, None),
        (None, [[], [10, 11, 12]]),
        (None, [[], [10, 11, 12]]),
    ]
    assert ak.to_list(ak.combinations(array, 2, axis=1)) == [
        [([0, 1, 2], []), ([0, 1, 2], [3, 4]), ([], [3, 4])],
        [],
        None,
        None,
        [([], [10, 11, 12])],
    ]
    assert ak.to_list(ak.combinations(array, 2, axis=-2)) == [
        [([0, 1, 2], []), ([0, 1, 2], [3, 4]), ([], [3, 4])],
        [],
        None,
        None,
        [([], [10, 11, 12])],
    ]
    assert ak.to_list(ak.combinations(array, 2, axis=2)) == [
        [[(0, 1), (0, 2), (1, 2)], [], [(3, 4)]],
        [],
        None,
        None,
        [[], [(10, 11), (10, 12), (11, 12)]],
    ]
    assert ak.to_list(ak.combinations(array, 2, axis=-1)) == [
        [[(0, 1), (0, 2), (1, 2)], [], [(3, 4)]],
        [],
        None,
        None,
        [[], [(10, 11), (10, 12), (11, 12)]],
    ]


def test_IndexedOptionArray_combinations():
    content = ak.from_iter(
        [[[0, 1, 2], [], [3, 4]], [], [[5]], [[6, 7, 8, 9]], [[], [10, 11, 12]]],
        highlevel=False,
    )
    index = ak.layout.Index64(np.array([0, 1, -1, -1, 4], dtype=np.int64))
    array = ak.Array(ak.layout.IndexedOptionArray64(index, content))
    assert ak.to_list(array) == [
        [[0, 1, 2], [], [3, 4]],
        [],
        None,
        None,
        [[], [10, 11, 12]],
    ]
    assert ak.to_list(ak.combinations(array, 2, axis=0)) == [
        ([[0, 1, 2], [], [3, 4]], []),
        ([[0, 1, 2], [], [3, 4]], None),
        ([[0, 1, 2], [], [3, 4]], None),
        ([[0, 1, 2], [], [3, 4]], [[], [10, 11, 12]]),
        ([], None),
        ([], None),
        ([], [[], [10, 11, 12]]),
        (None, None),
        (None, [[], [10, 11, 12]]),
        (None, [[], [10, 11, 12]]),
    ]
    assert ak.to_list(ak.combinations(array, 2, axis=-3)) == [
        ([[0, 1, 2], [], [3, 4]], []),
        ([[0, 1, 2], [], [3, 4]], None),
        ([[0, 1, 2], [], [3, 4]], None),
        ([[0, 1, 2], [], [3, 4]], [[], [10, 11, 12]]),
        ([], None),
        ([], None),
        ([], [[], [10, 11, 12]]),
        (None, None),
        (None, [[], [10, 11, 12]]),
        (None, [[], [10, 11, 12]]),
    ]
    assert ak.to_list(ak.combinations(array, 2, axis=1)) == [
        [([0, 1, 2], []), ([0, 1, 2], [3, 4]), ([], [3, 4])],
        [],
        None,
        None,
        [([], [10, 11, 12])],
    ]
    assert ak.to_list(ak.combinations(array, 2, axis=-2)) == [
        [([0, 1, 2], []), ([0, 1, 2], [3, 4]), ([], [3, 4])],
        [],
        None,
        None,
        [([], [10, 11, 12])],
    ]
    assert ak.to_list(ak.combinations(array, 2, axis=2)) == [
        [[(0, 1), (0, 2), (1, 2)], [], [(3, 4)]],
        [],
        None,
        None,
        [[], [(10, 11), (10, 12), (11, 12)]],
    ]
    assert ak.to_list(ak.combinations(array, 2, axis=-1)) == [
        [[(0, 1), (0, 2), (1, 2)], [], [(3, 4)]],
        [],
        None,
        None,
        [[], [(10, 11), (10, 12), (11, 12)]],
    ]


def test_merge():
    content = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array1 = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    assert ak.to_list(array1) == [[0.0, 1.1, 2.2], [], None, None, [6.6, 7.7, 8.8, 9.9]]
    array2 = ak.Array([[0.0, 1.1, 2.2], [], None, None, [6.6, 7.7, 8.8, 9.9]])
    array12 = ak.concatenate([array1, array2], highlevel=False)
    assert ak.to_list(array12) == [
        [0.0, 1.1, 2.2],
        [],
        None,
        None,
        [6.6, 7.7, 8.8, 9.9],
        [0.0, 1.1, 2.2],
        [],
        None,
        None,
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert isinstance(array12, ak.layout.IndexedOptionArray64)
    assert isinstance(
        array12.content, (ak.layout.ListArray64, ak.layout.ListOffsetArray64)
    )
    assert isinstance(array12.content.content, ak.layout.NumpyArray)
    assert ak.to_list(array12.content.content) == [
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
        0.0,
        1.1,
        2.2,
        6.6,
        7.7,
        8.8,
        9.9,
    ]


def test_BitMaskedArray():
    content = ak.layout.NumpyArray(np.arange(13))
    mask = ak.layout.IndexU8(np.array([58, 59], dtype=np.uint8))
    array = ak.layout.BitMaskedArray(
        mask, content, valid_when=False, length=13, lsb_order=False
    )
    assert np.asarray(array.bytemask()).tolist() == [
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
    assert ak.to_json(array) == "[0,1,null,null,null,5,null,7,8,9,null,null,null]"
    assert ak.to_list(array[1:-1]) == [
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
    ]
    assert ak.to_list(array[8:]) == [8, 9, None, None, None]

    array = ak.layout.BitMaskedArray(
        mask, content, valid_when=False, length=13, lsb_order=True
    )
    assert np.asarray(array.bytemask()).tolist() == [
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
    ]
    assert np.asarray(array.toByteMaskedArray().mask).tolist() == [
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
    ]
    assert np.asarray(array.toIndexedOptionArray64().index).tolist() == [
        0,
        -1,
        2,
        -1,
        -1,
        -1,
        6,
        7,
        -1,
        -1,
        10,
        -1,
        -1,
    ]
    assert ak.to_list(array) == [
        0,
        None,
        2,
        None,
        None,
        None,
        6,
        7,
        None,
        None,
        10,
        None,
        None,
    ]
    assert ak.to_json(array) == "[0,null,2,null,null,null,6,7,null,null,10,null,null]"
    assert ak.to_list(array[1:-1]) == [
        None,
        2,
        None,
        None,
        None,
        6,
        7,
        None,
        None,
        10,
        None,
    ]
    assert ak.to_list(array[8:]) == [None, None, 10, None, None]


def test_UnmaskedArray():
    content = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)
    )
    array = ak.layout.UnmaskedArray(content)
    assert ak.to_list(array) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert str(ak.type(content)) == "float64"
    assert str(ak.type(ak.Array(content))) == "5 * float64"
    assert str(ak.type(array)) == "?float64"
    assert str(ak.type(ak.Array(array))) == "5 * ?float64"


def test_tomask():
    array = ak.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]])
    mask1 = ak.Array([True, True, False, False, True])
    assert ak.to_list(array[mask1]) == [[0.0, 1.1, 2.2], [], [6.6, 7.7, 8.8, 9.9]]
    assert ak.to_list(ak.mask(array, mask1)) == [
        [0.0, 1.1, 2.2],
        [],
        None,
        None,
        [6.6, 7.7, 8.8, 9.9],
    ]

    mask2 = ak.Array(
        [[False, True, False], [], [True, True], [False], [True, False, False, True]]
    )
    assert ak.to_list(array[mask2]) == [[1.1], [], [3.3, 4.4], [], [6.6, 9.9]]
    assert ak.to_list(ak.mask(array, mask2)) == [
        [None, 1.1, None],
        [],
        [3.3, 4.4],
        [None],
        [6.6, None, None, 9.9],
    ]
