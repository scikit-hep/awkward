# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_keep_None_in_place_test():
    v2_array = ak.highlevel.Array([[3, 2, 1], [], None, [4, 5]]).layout

    assert to_list(ak.argsort(v2_array, axis=1, highlevel=False)) == [
        [2, 1, 0],
        [],
        None,
        [0, 1],
    ]

    assert to_list(ak.sort(v2_array, axis=1, highlevel=False)) == [
        [1, 2, 3],
        [],
        None,
        [4, 5],
    ]

    assert to_list(ak.sort(v2_array, axis=1, highlevel=False)) == [
        [1, 2, 3],
        [],
        None,
        [4, 5],
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), axis=1, highlevel=False).form
        == ak.argsort(v2_array, axis=1, highlevel=False).form
    )

    assert to_list(ak.argsort(v2_array, axis=1, highlevel=False)) == [
        [2, 1, 0],
        [],
        None,
        [0, 1],
    ]


def test_keep_None_in_place_test_2():
    v2_array = ak.highlevel.Array([[3, 2, 1], [], None, [4, 5]]).layout
    assert (
        ak.argsort(v2_array.to_typetracer(), axis=1, highlevel=False).form
        == ak.argsort(v2_array, axis=1, highlevel=False).form
    )


def test_empty_slice():
    electron = ak.highlevel.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 0, 1], np.int64)),
            ak.contents.RecordArray(
                [ak.contents.NumpyArray(np.array([1.0]))],
                ["pt"],
                parameters={"__record__": "Electron"},
            ),
        )
    )
    v2_electron = electron.layout[[[], []]]

    assert to_list(v2_electron) == [[], []]


def test_masked():
    v2_array = ak.highlevel.Array([[0, 1, 2, 3], [3, 3, 3, 2, 1]])
    is_valid = v2_array != 3

    v2_array_mask = ak.highlevel.Array(
        ak.contents.ListOffsetArray(
            v2_array.layout.offsets,
            ak.contents.ByteMaskedArray(
                ak.index.Index8(is_valid.layout.content.data),
                v2_array.layout.content,
                valid_when=True,
            ),
        )
    )

    assert to_list(v2_array_mask) == [
        [0, 1, 2, None],
        [None, None, None, 2, 1],
    ]

    assert to_list(ak.sort(v2_array_mask.layout, axis=1, highlevel=False)) == [
        [0, 1, 2, None],
        [1, 2, None, None, None],
    ]
    assert (
        ak.sort(v2_array_mask.layout.to_typetracer(), axis=1, highlevel=False).form
        == ak.sort(v2_array_mask.layout, axis=1, highlevel=False).form
    )


def test_v1_argsort_and_v2_sort():
    v2_array = ak.highlevel.Array([1, 2, None, 3, 0, None]).layout
    assert to_list(ak.sort(v2_array, highlevel=False)) == [
        0,
        1,
        2,
        3,
        None,
        None,
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )


def test_v1_argsort_2d_and_v2_sort():
    v2_array = ak.highlevel.Array(
        [[1, 2, None, 3, 0, None], [1, 2, None, 3, 0, None]]
    ).layout
    assert to_list(ak.sort(v2_array, highlevel=False)) == [
        [
            0,
            1,
            2,
            3,
            None,
            None,
        ],
        [
            0,
            1,
            2,
            3,
            None,
            None,
        ],
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )


def test_nan():
    v2_array = ak.highlevel.Array([1, 2, np.nan, 3, 0, np.nan]).layout
    assert (
        str(to_list(ak.sort(v2_array, highlevel=False)))
        == "[nan, nan, 0.0, 1.0, 2.0, 3.0]"
    )
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )


def test_sort_strings():
    v2_array = ak.highlevel.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    ).layout
    assert to_list(v2_array) == [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
    ]
    assert to_list(ak.sort(v2_array, highlevel=False)) == [
        "eight",
        "five",
        "four",
        "one",
        "seven",
        "six",
        "three",
        "two",
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )


def test_sort_nested_strings():
    v2_array = ak.highlevel.Array(
        [["one", "two"], ["three", "four", "five"], ["six"], ["seven", "eight"]]
    ).layout
    assert to_list(v2_array) == [
        ["one", "two"],
        ["three", "four", "five"],
        ["six"],
        ["seven", "eight"],
    ]
    assert to_list(ak.sort(v2_array, highlevel=False)) == [
        ["one", "two"],
        ["five", "four", "three"],
        ["six"],
        ["eight", "seven"],
    ]

    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )


def test_sort_invalid_axis():
    v2_array = ak.operations.from_numpy(
        np.array([[3.3, 2.2], [1.1, 5.5], [4.4, 6.6]]),
        regulararray=True,
        highlevel=False,
    )

    with pytest.raises(
        ValueError,
        match=r"axis=3 exceeds the depth of the nested list structure \(which is 2\)",
    ):
        ak.sort(v2_array, axis=3, highlevel=False)


def test_numpy_array_iscontiguous():
    matrix = np.arange(64).reshape(8, -1)
    v2_layout = ak.contents.NumpyArray(matrix[:, 0])

    assert not v2_layout.is_contiguous

    assert to_list(v2_layout) == [0, 8, 16, 24, 32, 40, 48, 56]

    matrix2 = np.arange(64).reshape(8, -1)
    v2_array = ak.contents.NumpyArray(matrix2[:, 0])
    assert not v2_array.is_contiguous

    assert to_list(ak.sort(v2_array, highlevel=False)) == [0, 8, 16, 24, 32, 40, 48, 56]
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )


def test_numpyarray_sort():
    v2_array = ak.operations.from_numpy(
        np.array([3.3, 2.2, 1.1, 5.5, 4.4]), regulararray=True, highlevel=False
    )
    assert to_list(np.sort(np.asarray(v2_array))) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert to_list(ak.sort(v2_array, highlevel=False)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )


def test_3d():
    array = ak.Array(
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

    assert to_list(
        ak.operations.argsort(array, axis=2, ascending=True, stable=False)
    ) == to_list(np.argsort(array, 2))
    assert to_list(
        ak.operations.sort(array, axis=2, ascending=True, stable=False)
    ) == to_list(np.sort(np.asarray(array), 2))
    assert to_list(
        ak.operations.argsort(array, axis=1, ascending=True, stable=False)
    ) == to_list(np.argsort(np.asarray(array), 1))
    assert to_list(
        ak.operations.sort(array, axis=1, ascending=True, stable=False)
    ) == to_list(np.sort(np.asarray(array), 1))
    assert to_list(
        ak.operations.sort(np.asarray(array), axis=1, ascending=False, stable=False)
    ) == [
        [
            [11.11, 12.12, 13.13, 14.14, 15.15],
            [6.6, 7.7, 8.8, 9.9, 10.1],
            [1.1, 2.2, 3.3, 4.4, 5.5],
        ],
        [
            [-1.1, -2.2, -3.3, -4.4, -5.5],
            [-6.6, -7.7, -8.8, -9.9, -10.1],
            [-11.11, -12.12, -13.13, -14.14, -15.15],
        ],
    ]
    assert to_list(
        ak.operations.sort(array, axis=0, ascending=True, stable=False)
    ) == to_list(np.sort(np.asarray(array), 0))
    assert to_list(
        ak.operations.argsort(array, axis=0, ascending=True, stable=False)
    ) == to_list(np.argsort(np.asarray(array), 0))


def test_bool_sort():
    v2_array = ak.operations.from_numpy(
        np.array([True, False, True, False, False]), regulararray=True, highlevel=False
    )
    assert to_list(ak.sort(v2_array, highlevel=False)) == [
        False,
        False,
        False,
        True,
        True,
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )


def test_emptyarray_sort():
    v2_array = ak.contents.emptyarray.EmptyArray()
    assert to_list(ak.sort(v2_array, highlevel=False)) == []

    v2_array = ak.highlevel.Array([[], [], []]).layout
    assert to_list(ak.sort(v2_array, highlevel=False)) == [[], [], []]
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )


def test_listarray_sort():
    v2_array = ak.contents.listarray.ListArray(
        ak.index.Index(np.array([4, 100, 1])),
        ak.index.Index(np.array([7, 100, 3, 200])),
        ak.contents.numpyarray.NumpyArray(
            np.array([6.6, 4.4, 5.5, 7.7, 3.3, 2.2, 1.1, 8.8])
        ),
    )

    assert to_list(v2_array) == [
        [3.3, 2.2, 1.1],
        [],
        [4.4, 5.5],
    ]

    assert to_list(ak.sort(v2_array, highlevel=False)) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )


def test_listoffsetarray_sort():
    v2_array = ak.operations.from_iter(
        [[3.3, 2.2, 1.1], [], [5.5, 4.4], [6.6], [9.9, 7.7, 8.8, 10.1]], highlevel=False
    )
    assert to_list(ak.sort(v2_array, highlevel=False)) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9, 10.10],
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )
    assert to_list(ak.sort(v2_array, axis=0, highlevel=False)) == [
        [3.3, 2.2, 1.1],
        [],
        [5.5, 4.4],
        [6.6],
        [9.9, 7.7, 8.8, 10.1],
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), axis=0, highlevel=False).form
        == ak.sort(v2_array, axis=0, highlevel=False).form
    )

    v2_array = ak.operations.from_iter(
        [
            [[11.1, 0.0, -2.2], [], [33.33, 4.4]],
            [],
            [[5.5]],
            [[6.6, -9.9, 8.8, 7.7]],
            [[], [12.2, 1.1, 10.0]],
        ],
        highlevel=False,
    )
    assert to_list(ak.sort(v2_array, axis=0, highlevel=False)) == [
        [[5.5, -9.9, -2.2], [], [33.33, 4.4]],
        [],
        [[6.6]],
        [[11.1, 0.0, 8.8, 7.7]],
        [[], [12.2, 1.1, 10.0]],
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), axis=0, highlevel=False).form
        == ak.sort(v2_array, axis=0, highlevel=False).form
    )
    # [
    #     [[5.5, -9.9, -2.2], [], [33.33, 4.4]],
    #     [],
    #     [[6.6]],
    #     [[11.1, 0.0, 8.8, 7.7]],
    #     [[], [12.2, 1.1, 10.0]]
    # ]
    assert to_list(ak.sort(v2_array, axis=1, highlevel=False)) == [
        [[11.1, 0.0, -2.2], [], [33.33, 4.4]],
        [],
        [[5.5]],
        [[6.6, -9.9, 8.8, 7.7]],
        [[], [12.2, 1.1, 10.0]],
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), axis=1, highlevel=False).form
        == ak.sort(v2_array, axis=1, highlevel=False).form
    )
    # [
    #     [[11.1, 0.0, -2.2], [], [33.33, 4.4]],
    #     [],
    #     [[5.5]],
    #     [[6.6, -9.9, 8.8, 7.7]],
    #     [[], [12.2, 1.1, 10.0]]
    # ]
    assert to_list(ak.sort(v2_array, axis=2, highlevel=False)) == [
        [[-2.2, 0.0, 11.1], [], [4.4, 33.33]],
        [],
        [[5.5]],
        [[-9.9, 6.6, 7.7, 8.8]],
        [[], [1.1, 10.0, 12.2]],
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), axis=2, highlevel=False).form
        == ak.sort(v2_array, axis=2, highlevel=False).form
    )
    # [
    #     [[-2.2, 0.0, 11.1], [], [4.4, 33.33]],
    #     [],
    #     [[5.5]],
    #     [[-9.9, 6.6, 7.7, 8.8]],
    #     [[], [1.1, 10.0, 12.2]]
    # ]
    assert to_list(ak.sort(v2_array, highlevel=False)) == [
        [[-2.2, 0.0, 11.1], [], [4.4, 33.33]],
        [],
        [[5.5]],
        [[-9.9, 6.6, 7.7, 8.8]],
        [[], [1.1, 10.0, 12.2]],
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )
    # [
    #     [[-2.2, 0.0, 11.1], [], [4.4, 33.33]],
    #     [],
    #     [[5.5]],
    #     [[-9.9, 6.6, 7.7, 8.8]],
    #     [[], [1.1, 10.0, 12.2]]
    # ]


def test_regulararray_sort():
    v2_array = ak.operations.from_numpy(
        np.array(
            [
                [
                    [3.3, 1.1, 5.5, 2.2, 4.4],
                    [8.8, 6.6, 9.9, 7.7, 10.10],
                    [11.11, 14.14, 15.15, 12.12, 13.13],
                ],
                [
                    [-1.1, -2.2, -5.5, -3.3, -4.4],
                    [-7.7, -8.8, -9.9, -6.6, -10.1],
                    [-13.13, -11.11, -12.12, -14.14, -15.15],
                ],
            ]
        ),
        regulararray=True,
        highlevel=False,
    )
    assert to_list(v2_array) == [
        [
            [3.3, 1.1, 5.5, 2.2, 4.4],
            [8.8, 6.6, 9.9, 7.7, 10.1],
            [11.11, 14.14, 15.15, 12.12, 13.13],
        ],
        [
            [-1.1, -2.2, -5.5, -3.3, -4.4],
            [-7.7, -8.8, -9.9, -6.6, -10.1],
            [-13.13, -11.11, -12.12, -14.14, -15.15],
        ],
    ]
    assert to_list(ak.sort(v2_array, highlevel=False)) == [
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
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )


def test_bytemaskedarray_sort():
    content = ak.operations.from_iter(
        [
            [[1.1, 0.0, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 9.9, 8.8, 7.7]],
            [[], [12.2, 11.1, 10.0]],
        ],
        highlevel=False,
    )
    mask = ak.index.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    v2_array = ak.contents.ByteMaskedArray(mask, content, valid_when=False)

    assert to_list(v2_array) == [
        [[1.1, 0.0, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [12.2, 11.1, 10.0]],
    ]
    assert to_list(ak.sort(v2_array, highlevel=False)) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )


def test_bytemaskedarray_sort_2():
    array3 = ak.highlevel.Array(
        [[2.2, 1.1, 3.3], [], [4.4, 5.5], [5.5], [-4.4, -5.5, -6.6]]
    ).layout

    assert to_list(
        ak.operations.sort(array3, axis=1, ascending=False, stable=False)
    ) == [
        [3.3, 2.2, 1.1],
        [],
        [5.5, 4.4],
        [5.5],
        [-4.4, -5.5, -6.6],
    ]

    assert to_list(
        ak.operations.sort(array3, axis=0, ascending=True, stable=False)
    ) == [
        [-4.4, -5.5, -6.6],
        [],
        [2.2, 1.1],
        [4.4],
        [5.5, 5.5, 3.3],
    ]

    content = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.index.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.contents.ByteMaskedArray(mask, content, valid_when=False)
    assert to_list(
        ak.operations.argsort(array, axis=0, ascending=True, stable=False)
    ) == [
        [0, 0, 0],
        [],
        [2, 2, 2, 2],
        None,
        None,
    ]

    assert to_list(ak.operations.sort(array, axis=0, ascending=True, stable=False)) == [
        [0.0, 1.1, 2.2],
        [],
        [6.6, 7.7, 8.8, 9.9],
        None,
        None,
    ]

    assert to_list(
        ak.operations.sort(array, axis=0, ascending=False, stable=False)
    ) == [
        [6.6, 7.7, 8.8],
        [],
        [0.0, 1.1, 2.2, 9.9],
        None,
        None,
    ]

    assert to_list(
        ak.operations.argsort(array, axis=1, ascending=True, stable=False)
    ) == [
        [0, 1, 2],
        [],
        None,
        None,
        [0, 1, 2, 3],
    ]

    assert to_list(ak.operations.sort(array, 1, ascending=False, stable=False)) == [
        [2.2, 1.1, 0.0],
        [],
        None,
        None,
        [9.9, 8.8, 7.7, 6.6],
    ]


def test_bitmaskedarray_sort():
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

    assert to_list(ak.sort(v2_array, highlevel=False)) == [
        0.0,
        1.0,
        1.1,
        2.0,
        3.0,
        3.3,
        5.5,
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )


def test_unmaskedarray_sort():
    v2_array = ak.contents.unmaskedarray.UnmaskedArray(
        ak.contents.numpyarray.NumpyArray(
            np.array([0.0, 2.2, 1.1, 3.3], dtype=np.float64)
        )
    )
    assert to_list(ak.sort(v2_array, highlevel=False)) == [0.0, 1.1, 2.2, 3.3]
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )


@pytest.mark.skip(
    reason="I can't think of a canonical UnionArray (non-mergeable contents) that can be used in sorting"
)
def test_unionarray_sort():
    v2_array = ak.contents.unionarray.UnionArray(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak.contents.numpyarray.NumpyArray(np.array([1, 2, 3])),
            ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )

    assert to_list(v2_array) == [5.5, 4.4, 1, 2, 3.3, 3, 5.5]
    assert to_list(ak.sort(v2_array, highlevel=False)) == [
        1.0,
        2.0,
        3.0,
        3.3,
        4.4,
        5.5,
        5.5,
    ]


@pytest.mark.skip(
    reason="I can't think of a canonical UnionArray (non-mergeable contents) that can be used in sorting"
)
def test_unionarray_sort_2():
    v2_array = ak.contents.unionarray.UnionArray(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak.contents.numpyarray.NumpyArray(np.array([1, 2, 3])),
            ak.contents.numpyarray.NumpyArray(np.array([7, 0, 3, 4, 5])),
        ],
    )

    assert to_list(v2_array) == [5, 4, 1, 2, 3, 3, 5]
    assert to_list(ak.sort(v2_array, highlevel=False)) == [1, 2, 3, 3, 4, 5, 5]


def test_indexedarray_sort():
    v2_array = ak.contents.indexedarray.IndexedArray(
        ak.index.Index(np.array([2, 2, 0, 1, 4, 5, 4])),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    assert to_list(v2_array) == [3.3, 3.3, 1.1, 2.2, 5.5, 6.6, 5.5]
    assert to_list(ak.sort(v2_array, highlevel=False)) == [
        1.1,
        2.2,
        3.3,
        3.3,
        5.5,
        5.5,
        6.6,
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )


def test_indexedoptionarray_sort():
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
    assert to_list(ak.sort(v2_array, highlevel=False)) == [
        {"nest": 2.2},
        {"nest": 3.3},
        {"nest": 3.3},
        {"nest": 5.5},
        {"nest": 6.6},
        None,
        None,
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )

    v2_array = ak.highlevel.Array(
        [[1, 2, None, 3, 0, None], [1, 2, None, 3, 0, None]]
    ).layout
    assert to_list(v2_array) == [
        [1, 2, None, 3, 0, None],
        [1, 2, None, 3, 0, None],
    ]
    assert to_list(ak.sort(v2_array, highlevel=False)) == [
        [0, 1, 2, 3, None, None],
        [0, 1, 2, 3, None, None],
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )

    v2_array = ak.highlevel.Array(
        [
            [None, None, 2.2, 1.1, 3.3],
            [None, None, None],
            [4.4, None, 5.5],
            [5.5, None, None],
            [-4.4, -5.5, -6.6],
        ]
    ).layout

    assert to_list(ak.sort(v2_array, highlevel=False)) == [
        [1.1, 2.2, 3.3, None, None],
        [None, None, None],
        [4.4, 5.5, None],
        [5.5, None, None],
        [-6.6, -5.5, -4.4],
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )

    assert to_list(ak.sort(v2_array, axis=0, highlevel=False)) == [
        [-4.4, -5.5, -6.6, 1.1, 3.3],
        [4.4, None, 2.2],
        [5.5, None, 5.5],
        [None, None, None],
        [None, None, None],
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), axis=0, highlevel=False).form
        == ak.sort(v2_array, axis=0, highlevel=False).form
    )

    assert to_list(
        ak.sort(v2_array, axis=1, ascending=True, stable=False, highlevel=False)
    ) == [
        [1.1, 2.2, 3.3, None, None],
        [None, None, None],
        [4.4, 5.5, None],
        [5.5, None, None],
        [-6.6, -5.5, -4.4],
    ]
    assert (
        ak.sort(
            v2_array.to_typetracer(),
            axis=1,
            ascending=True,
            stable=False,
            highlevel=False,
        ).form
        == ak.sort(v2_array, axis=1, ascending=True, stable=False, highlevel=False).form
    )

    assert to_list(
        ak.sort(v2_array, axis=1, ascending=False, stable=True, highlevel=False)
    ) == [
        [3.3, 2.2, 1.1, None, None],
        [None, None, None],
        [5.5, 4.4, None],
        [5.5, None, None],
        [-4.4, -5.5, -6.6],
    ]
    assert (
        ak.sort(
            v2_array.to_typetracer(),
            axis=1,
            ascending=False,
            stable=True,
            highlevel=False,
        ).form
        == ak.sort(v2_array, axis=1, ascending=False, stable=True, highlevel=False).form
    )

    assert to_list(
        ak.sort(v2_array, axis=1, ascending=False, stable=False, highlevel=False)
    ) == [
        [3.3, 2.2, 1.1, None, None],
        [None, None, None],
        [5.5, 4.4, None],
        [5.5, None, None],
        [-4.4, -5.5, -6.6],
    ]
    assert (
        ak.sort(
            v2_array.to_typetracer(),
            axis=1,
            ascending=False,
            stable=False,
            highlevel=False,
        ).form
        == ak.sort(
            v2_array, axis=1, ascending=False, stable=False, highlevel=False
        ).form
    )


def test_sort_zero_length_arrays():
    array = ak.contents.IndexedArray(
        ak.index.Index64([]), ak.contents.NumpyArray([1, 2, 3])
    )
    assert to_list(array) == []
    assert to_list(ak.operations.sort(array)) == []
    assert to_list(ak.operations.argsort(array)) == []

    content0 = ak.operations.from_iter(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False
    )
    content1 = ak.operations.from_iter(
        ["one", "two", "three", "four", "five"], highlevel=False
    )
    tags = ak.index.Index8([])
    index = ak.index.Index32([])
    array = ak.contents.UnionArray(tags, index, [content0, content1])
    assert to_list(array) == []
    assert to_list(ak.operations.sort(array)) == []
    assert to_list(ak.operations.argsort(array)) == []

    content = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.index.Index8([])
    array = ak.contents.ByteMaskedArray(mask, content, valid_when=False)
    assert to_list(array) == []
    assert to_list(ak.operations.sort(array)) == []
    assert to_list(ak.operations.argsort(array)) == []

    array = ak.contents.NumpyArray([])
    assert to_list(array) == []
    assert to_list(ak.operations.sort(array)) == []
    assert to_list(ak.operations.argsort(array)) == []

    array = ak.contents.RecordArray([], None, 0)
    assert to_list(array) == []
    assert to_list(ak.operations.sort(array)) == []

    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts1 = ak.index.Index64([])
    stops1 = ak.index.Index64([])
    offsets1 = ak.index.Index64(np.array([0]))
    array = ak.contents.ListArray(starts1, stops1, content)
    assert to_list(array) == []
    assert to_list(ak.operations.sort(array)) == []
    assert to_list(ak.operations.argsort(array)) == []

    array = ak.contents.ListOffsetArray(offsets1, content)
    assert to_list(array) == []
    assert to_list(ak.operations.sort(array)) == []
    assert to_list(ak.operations.argsort(array)) == []


def test_recordarray_sort():
    v2_array = ak.contents.regulararray.RegularArray(
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([0.0, 1.1, 2.2, 33.33, 4.4, 5.5, -6.6])
                )
            ],
            ["nest"],
        ),
        3,
    )
    assert to_list(ak.sort(v2_array, highlevel=False)) == [
        [{"nest": 0.0}, {"nest": 1.1}, {"nest": 2.2}],
        [{"nest": 4.4}, {"nest": 5.5}, {"nest": 33.33}],
    ]
    assert (
        ak.sort(v2_array.to_typetracer(), highlevel=False).form
        == ak.sort(v2_array, highlevel=False).form
    )
