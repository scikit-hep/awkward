# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test_keep_None_in_place_test():
    v2_array = ak._v2.highlevel.Array([[3, 2, 1], [], None, [4, 5]]).layout

    assert to_list(v2_array.argsort(axis=1)) == [
        [2, 1, 0],
        [],
        None,
        [0, 1],
    ]

    assert to_list(v2_array.sort(axis=1)) == [
        [1, 2, 3],
        [],
        None,
        [4, 5],
    ]

    assert to_list(v2_array.sort(axis=1)) == [[1, 2, 3], [], None, [4, 5]]
    assert v2_array.typetracer.sort(axis=1).form == v2_array.argsort(axis=1).form

    assert to_list(v2_array.argsort(axis=1)) == [[2, 1, 0], [], None, [0, 1]]


def test_keep_None_in_place_test_2():
    v2_array = ak._v2.highlevel.Array([[3, 2, 1], [], None, [4, 5]]).layout
    assert v2_array.typetracer.argsort(axis=1).form == v2_array.argsort(axis=1).form


def test_empty_slice():
    electron = ak._v2.highlevel.Array(
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([0, 0, 1], np.int64)),
            ak._v2.contents.RecordArray(
                [ak._v2.contents.NumpyArray(np.array([1.0]))],
                ["pt"],
                parameters={"__record__": "Electron"},
            ),
        )
    )
    v2_electron = electron.layout[[[], []]]

    assert to_list(v2_electron) == [[], []]


def test_masked():
    v2_array = ak._v2.highlevel.Array([[0, 1, 2, 3], [3, 3, 3, 2, 1]])
    is_valid = v2_array != 3

    v2_array_mask = ak._v2.highlevel.Array(
        ak._v2.contents.ListOffsetArray(
            v2_array.layout.offsets,
            ak._v2.contents.ByteMaskedArray(
                ak._v2.index.Index8(is_valid.layout.content.data),
                v2_array.layout.content,
                valid_when=True,
            ),
        )
    )

    assert to_list(v2_array_mask) == [
        [0, 1, 2, None],
        [None, None, None, 2, 1],
    ]

    assert to_list(v2_array_mask.layout.sort(axis=1)) == [
        [0, 1, 2, None],
        [1, 2, None, None, None],
    ]
    assert (
        v2_array_mask.layout.typetracer.sort(axis=1).form
        == v2_array_mask.layout.sort(axis=1).form
    )


def test_v1_argsort_and_v2_sort():
    v2_array = ak._v2.highlevel.Array([1, 2, None, 3, 0, None]).layout
    assert to_list(v2_array.sort()) == [
        0,
        1,
        2,
        3,
        None,
        None,
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_v1_argsort_2d_and_v2_sort():
    v2_array = ak._v2.highlevel.Array(
        [[1, 2, None, 3, 0, None], [1, 2, None, 3, 0, None]]
    ).layout
    assert to_list(v2_array.sort()) == [
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
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_nan():
    v2_array = ak._v2.highlevel.Array([1, 2, np.nan, 3, 0, np.nan]).layout
    assert str(to_list(v2_array.sort())) == "[nan, nan, 0.0, 1.0, 2.0, 3.0]"
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_sort_strings():
    v2_array = ak._v2.highlevel.Array(
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
    assert to_list(v2_array.sort()) == [
        "eight",
        "five",
        "four",
        "one",
        "seven",
        "six",
        "three",
        "two",
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_sort_nested_strings():
    v2_array = ak._v2.highlevel.Array(
        [["one", "two"], ["three", "four", "five"], ["six"], ["seven", "eight"]]
    ).layout
    assert to_list(v2_array) == [
        ["one", "two"],
        ["three", "four", "five"],
        ["six"],
        ["seven", "eight"],
    ]
    assert to_list(v2_array.sort()) == [
        ["one", "two"],
        ["five", "four", "three"],
        ["six"],
        ["eight", "seven"],
    ]

    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_sort_invalid_axis():
    v2_array = ak._v2.operations.from_numpy(
        np.array([[3.3, 2.2], [1.1, 5.5], [4.4, 6.6]]),
        regulararray=True,
        highlevel=False,
    )

    with pytest.raises(ValueError) as err:
        v2_array.sort(axis=3)
    assert str(err.value).startswith(
        "axis=3 exceeds the depth of the nested list structure (which is 2)"
    )


def test_numpy_array_iscontiguous():
    matrix = np.arange(64).reshape(8, -1)
    v2_layout = ak._v2.contents.NumpyArray(matrix[:, 0])

    assert not v2_layout.is_contiguous

    assert to_list(v2_layout) == [0, 8, 16, 24, 32, 40, 48, 56]

    matrix2 = np.arange(64).reshape(8, -1)
    v2_array = ak._v2.contents.NumpyArray(matrix2[:, 0])
    assert not v2_array.is_contiguous

    assert to_list(v2_array.sort()) == [0, 8, 16, 24, 32, 40, 48, 56]
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_numpyarray_sort():
    v2_array = ak._v2.operations.from_numpy(
        np.array([3.3, 2.2, 1.1, 5.5, 4.4]), regulararray=True, highlevel=False
    )
    assert to_list(np.sort(np.asarray(v2_array))) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert to_list(v2_array.sort()) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_3d():
    array = ak._v2.Array(
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
        ak._v2.operations.argsort(array, axis=2, ascending=True, stable=False)
    ) == to_list(np.argsort(array, 2))
    assert to_list(
        ak._v2.operations.sort(array, axis=2, ascending=True, stable=False)
    ) == to_list(np.sort(np.asarray(array), 2))
    assert to_list(
        ak._v2.operations.argsort(array, axis=1, ascending=True, stable=False)
    ) == to_list(np.argsort(np.asarray(array), 1))
    assert to_list(
        ak._v2.operations.sort(array, axis=1, ascending=True, stable=False)
    ) == to_list(np.sort(np.asarray(array), 1))
    assert to_list(
        ak._v2.operations.sort(np.asarray(array), axis=1, ascending=False, stable=False)
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
        ak._v2.operations.sort(array, axis=0, ascending=True, stable=False)
    ) == to_list(np.sort(np.asarray(array), 0))
    assert to_list(
        ak._v2.operations.argsort(array, axis=0, ascending=True, stable=False)
    ) == to_list(np.argsort(np.asarray(array), 0))


def test_bool_sort():
    v2_array = ak._v2.operations.from_numpy(
        np.array([True, False, True, False, False]), regulararray=True, highlevel=False
    )
    assert to_list(v2_array.sort()) == [
        False,
        False,
        False,
        True,
        True,
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_emptyarray_sort():
    v2_array = ak._v2.contents.emptyarray.EmptyArray()
    assert to_list(v2_array.sort()) == []

    v2_array = ak._v2.highlevel.Array([[], [], []]).layout
    assert to_list(v2_array.sort()) == [[], [], []]
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_listarray_sort():
    v2_array = ak._v2.contents.listarray.ListArray(  # noqa: F841
        ak._v2.index.Index(np.array([4, 100, 1])),
        ak._v2.index.Index(np.array([7, 100, 3, 200])),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array([6.6, 4.4, 5.5, 7.7, 3.3, 2.2, 1.1, 8.8])
        ),
    )

    assert to_list(v2_array) == [
        [3.3, 2.2, 1.1],
        [],
        [4.4, 5.5],
    ]

    assert to_list(v2_array.sort()) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_listoffsetarray_sort():
    v2_array = ak._v2.operations.from_iter(
        [[3.3, 2.2, 1.1], [], [5.5, 4.4], [6.6], [9.9, 7.7, 8.8, 10.1]], highlevel=False
    )
    assert to_list(v2_array.sort()) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9, 10.10],
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form
    assert to_list(v2_array.sort(axis=0)) == [
        [3.3, 2.2, 1.1],
        [],
        [5.5, 4.4],
        [6.6],
        [9.9, 7.7, 8.8, 10.1],
    ]
    assert v2_array.typetracer.sort(axis=0).form == v2_array.sort(axis=0).form

    v2_array = ak._v2.operations.from_iter(
        [
            [[11.1, 0.0, -2.2], [], [33.33, 4.4]],
            [],
            [[5.5]],
            [[6.6, -9.9, 8.8, 7.7]],
            [[], [12.2, 1.1, 10.0]],
        ],
        highlevel=False,
    )
    assert to_list(v2_array.sort(axis=0)) == [
        [[5.5, -9.9, -2.2], [], [33.33, 4.4]],
        [],
        [[6.6]],
        [[11.1, 0.0, 8.8, 7.7]],
        [[], [12.2, 1.1, 10.0]],
    ]
    assert v2_array.typetracer.sort(axis=0).form == v2_array.sort(axis=0).form
    # [
    #     [[5.5, -9.9, -2.2], [], [33.33, 4.4]],
    #     [],
    #     [[6.6]],
    #     [[11.1, 0.0, 8.8, 7.7]],
    #     [[], [12.2, 1.1, 10.0]]
    # ]
    assert to_list(v2_array.sort(axis=1)) == [
        [[11.1, 0.0, -2.2], [], [33.33, 4.4]],
        [],
        [[5.5]],
        [[6.6, -9.9, 8.8, 7.7]],
        [[], [12.2, 1.1, 10.0]],
    ]
    assert v2_array.typetracer.sort(axis=1).form == v2_array.sort(axis=1).form
    # [
    #     [[11.1, 0.0, -2.2], [], [33.33, 4.4]],
    #     [],
    #     [[5.5]],
    #     [[6.6, -9.9, 8.8, 7.7]],
    #     [[], [12.2, 1.1, 10.0]]
    # ]
    assert to_list(v2_array.sort(axis=2)) == [
        [[-2.2, 0.0, 11.1], [], [4.4, 33.33]],
        [],
        [[5.5]],
        [[-9.9, 6.6, 7.7, 8.8]],
        [[], [1.1, 10.0, 12.2]],
    ]
    assert v2_array.typetracer.sort(axis=2).form == v2_array.sort(axis=2).form
    # [
    #     [[-2.2, 0.0, 11.1], [], [4.4, 33.33]],
    #     [],
    #     [[5.5]],
    #     [[-9.9, 6.6, 7.7, 8.8]],
    #     [[], [1.1, 10.0, 12.2]]
    # ]
    assert to_list(v2_array.sort()) == [
        [[-2.2, 0.0, 11.1], [], [4.4, 33.33]],
        [],
        [[5.5]],
        [[-9.9, 6.6, 7.7, 8.8]],
        [[], [1.1, 10.0, 12.2]],
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form
    # [
    #     [[-2.2, 0.0, 11.1], [], [4.4, 33.33]],
    #     [],
    #     [[5.5]],
    #     [[-9.9, 6.6, 7.7, 8.8]],
    #     [[], [1.1, 10.0, 12.2]]
    # ]


def test_regulararray_sort():
    v2_array = ak._v2.operations.from_numpy(
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
    assert to_list(v2_array.sort()) == [
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
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_bytemaskedarray_sort():
    content = ak._v2.operations.from_iter(
        [
            [[1.1, 0.0, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 9.9, 8.8, 7.7]],
            [[], [12.2, 11.1, 10.0]],
        ],
        highlevel=False,
    )
    mask = ak._v2.index.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    v2_array = ak._v2.contents.ByteMaskedArray(mask, content, valid_when=False)

    assert to_list(v2_array) == [
        [[1.1, 0.0, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [12.2, 11.1, 10.0]],
    ]
    assert to_list(v2_array.sort()) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_bytemaskedarray_sort_2():
    array3 = ak._v2.highlevel.Array(
        [[2.2, 1.1, 3.3], [], [4.4, 5.5], [5.5], [-4.4, -5.5, -6.6]]
    ).layout

    assert to_list(
        ak._v2.operations.sort(array3, axis=1, ascending=False, stable=False)
    ) == [
        [3.3, 2.2, 1.1],
        [],
        [5.5, 4.4],
        [5.5],
        [-4.4, -5.5, -6.6],
    ]

    assert to_list(
        ak._v2.operations.sort(array3, axis=0, ascending=True, stable=False)
    ) == [
        [-4.4, -5.5, -6.6],
        [],
        [2.2, 1.1],
        [4.4],
        [5.5, 5.5, 3.3],
    ]

    content = ak._v2.operations.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak._v2.index.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak._v2.contents.ByteMaskedArray(mask, content, valid_when=False)
    assert to_list(
        ak._v2.operations.argsort(array, axis=0, ascending=True, stable=False)
    ) == [
        [0, 0, 0],
        [],
        [2, 2, 2, 2],
        None,
        None,
    ]

    assert to_list(
        ak._v2.operations.sort(array, axis=0, ascending=True, stable=False)
    ) == [
        [0.0, 1.1, 2.2],
        [],
        [6.6, 7.7, 8.8, 9.9],
        None,
        None,
    ]

    assert to_list(
        ak._v2.operations.sort(array, axis=0, ascending=False, stable=False)
    ) == [
        [6.6, 7.7, 8.8],
        [],
        [0.0, 1.1, 2.2, 9.9],
        None,
        None,
    ]

    assert to_list(
        ak._v2.operations.argsort(array, axis=1, ascending=True, stable=False)
    ) == [
        [0, 1, 2],
        [],
        None,
        None,
        [0, 1, 2, 3],
    ]

    assert to_list(ak._v2.operations.sort(array, 1, False, False)) == [
        [2.2, 1.1, 0.0],
        [],
        None,
        None,
        [9.9, 8.8, 7.7, 6.6],
    ]


def test_bitmaskedarray_sort():
    v2_array = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
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
        ak._v2.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )

    assert to_list(v2_array.sort()) == [
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
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_unmaskedarray_sort():
    v2_array = ak._v2.contents.unmaskedarray.UnmaskedArray(
        ak._v2.contents.numpyarray.NumpyArray(
            np.array([0.0, 2.2, 1.1, 3.3], dtype=np.float64)
        )
    )
    assert to_list(v2_array.sort()) == [0.0, 1.1, 2.2, 3.3]
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_unionarray_sort():
    v2_array = ak._v2.contents.unionarray.UnionArray(
        ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak._v2.contents.numpyarray.NumpyArray(np.array([1, 2, 3])),
            ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )

    assert to_list(v2_array) == [5.5, 4.4, 1, 2, 3.3, 3, 5.5]
    assert to_list(v2_array.sort()) == [1.0, 2.0, 3.0, 3.3, 4.4, 5.5, 5.5]


def test_unionarray_sort_2():
    v2_array = ak._v2.contents.unionarray.UnionArray(  # noqa: F841
        ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak._v2.contents.numpyarray.NumpyArray(np.array([1, 2, 3])),
            ak._v2.contents.numpyarray.NumpyArray(np.array([7, 0, 3, 4, 5])),
        ],
    )

    assert to_list(v2_array) == [5, 4, 1, 2, 3, 3, 5]
    assert to_list(v2_array.sort()) == [1, 2, 3, 3, 4, 5, 5]


def test_indexedarray_sort():
    v2_array = ak._v2.contents.indexedarray.IndexedArray(  # noqa: F841
        ak._v2.index.Index(np.array([2, 2, 0, 1, 4, 5, 4])),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    assert to_list(v2_array) == [3.3, 3.3, 1.1, 2.2, 5.5, 6.6, 5.5]
    assert to_list(v2_array.sort()) == [1.1, 2.2, 3.3, 3.3, 5.5, 5.5, 6.6]
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_indexedoptionarray_sort():
    v2_array = ak._v2.contents.indexedoptionarray.IndexedOptionArray(  # noqa: F841
        ak._v2.index.Index(np.array([2, 2, -1, 1, -1, 5, 4])),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
    )
    assert to_list(v2_array.sort()) == [
        {"nest": 2.2},
        {"nest": 3.3},
        {"nest": 3.3},
        {"nest": 5.5},
        {"nest": 6.6},
        None,
        None,
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form

    v2_array = ak._v2.highlevel.Array(
        [[1, 2, None, 3, 0, None], [1, 2, None, 3, 0, None]]
    ).layout
    assert to_list(v2_array) == [
        [1, 2, None, 3, 0, None],
        [1, 2, None, 3, 0, None],
    ]
    assert to_list(v2_array.sort()) == [
        [0, 1, 2, 3, None, None],
        [0, 1, 2, 3, None, None],
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form

    v2_array = ak._v2.highlevel.Array(
        [
            [None, None, 2.2, 1.1, 3.3],
            [None, None, None],
            [4.4, None, 5.5],
            [5.5, None, None],
            [-4.4, -5.5, -6.6],
        ]
    ).layout

    assert to_list(v2_array.sort()) == [
        [1.1, 2.2, 3.3, None, None],
        [None, None, None],
        [4.4, 5.5, None],
        [5.5, None, None],
        [-6.6, -5.5, -4.4],
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form

    assert to_list(v2_array.sort(axis=0)) == [
        [-4.4, -5.5, -6.6, 1.1, 3.3],
        [4.4, None, 2.2],
        [5.5, None, 5.5],
        [None, None, None],
        [None, None, None],
    ]
    assert v2_array.typetracer.sort(axis=0).form == v2_array.sort(axis=0).form

    assert to_list(v2_array.sort(axis=1, ascending=True, stable=False)) == [
        [1.1, 2.2, 3.3, None, None],
        [None, None, None],
        [4.4, 5.5, None],
        [5.5, None, None],
        [-6.6, -5.5, -4.4],
    ]
    assert (
        v2_array.typetracer.sort(axis=1, ascending=True, stable=False).form
        == v2_array.sort(axis=1, ascending=True, stable=False).form
    )

    assert to_list(v2_array.sort(axis=1, ascending=False, stable=True)) == [
        [3.3, 2.2, 1.1, None, None],
        [None, None, None],
        [5.5, 4.4, None],
        [5.5, None, None],
        [-4.4, -5.5, -6.6],
    ]
    assert (
        v2_array.typetracer.sort(axis=1, ascending=False, stable=True).form
        == v2_array.sort(axis=1, ascending=False, stable=True).form
    )

    assert to_list(v2_array.sort(axis=1, ascending=False, stable=False)) == [
        [3.3, 2.2, 1.1, None, None],
        [None, None, None],
        [5.5, 4.4, None],
        [5.5, None, None],
        [-4.4, -5.5, -6.6],
    ]
    assert (
        v2_array.typetracer.sort(axis=1, ascending=False, stable=False).form
        == v2_array.sort(axis=1, ascending=False, stable=False).form
    )


def test_sort_zero_length_arrays():
    array = ak._v2.contents.IndexedArray(
        ak._v2.index.Index64([]), ak._v2.contents.NumpyArray([1, 2, 3])
    )
    assert to_list(array) == []
    assert to_list(ak._v2.operations.sort(array)) == []
    assert to_list(ak._v2.operations.argsort(array)) == []

    content0 = ak._v2.operations.from_iter(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False
    )
    content1 = ak._v2.operations.from_iter(
        ["one", "two", "three", "four", "five"], highlevel=False
    )
    tags = ak._v2.index.Index8([])
    index = ak._v2.index.Index32([])
    array = ak._v2.contents.UnionArray(tags, index, [content0, content1])
    assert to_list(array) == []
    assert to_list(ak._v2.operations.sort(array)) == []
    assert to_list(ak._v2.operations.argsort(array)) == []

    content = ak._v2.operations.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak._v2.index.Index8([])
    array = ak._v2.contents.ByteMaskedArray(mask, content, valid_when=False)
    assert to_list(array) == []
    assert to_list(ak._v2.operations.sort(array)) == []
    assert to_list(ak._v2.operations.argsort(array)) == []

    array = ak._v2.contents.NumpyArray([])
    assert to_list(array) == []
    assert to_list(ak._v2.operations.sort(array)) == []
    assert to_list(ak._v2.operations.argsort(array)) == []

    array = ak._v2.contents.RecordArray([], None, 0)
    assert to_list(array) == []
    assert to_list(ak._v2.operations.sort(array)) == []

    content = ak._v2.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts1 = ak._v2.index.Index64([])
    stops1 = ak._v2.index.Index64([])
    offsets1 = ak._v2.index.Index64(np.array([0]))
    array = ak._v2.contents.ListArray(starts1, stops1, content)
    assert to_list(array) == []
    assert to_list(ak._v2.operations.sort(array)) == []
    assert to_list(ak._v2.operations.argsort(array)) == []

    array = ak._v2.contents.ListOffsetArray(offsets1, content)
    assert to_list(array) == []
    assert to_list(ak._v2.operations.sort(array)) == []
    assert to_list(ak._v2.operations.argsort(array)) == []


def test_recordarray_sort():
    v2_array = ak._v2.contents.regulararray.RegularArray(  # noqa: F841
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([0.0, 1.1, 2.2, 33.33, 4.4, 5.5, -6.6])
                )
            ],
            ["nest"],
        ),
        3,
    )
    assert to_list(v2_array.sort()) == [
        [{"nest": 0.0}, {"nest": 1.1}, {"nest": 2.2}],
        [{"nest": 4.4}, {"nest": 5.5}, {"nest": 33.33}],
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form
