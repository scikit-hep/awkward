# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_keep_None_in_place_test():
    v1_array = ak.Array([[3, 2, 1], [], None, [4, 5]])

    assert ak.to_list(ak.argsort(v1_array, axis=1)) == [
        [2, 1, 0],
        [],
        None,
        [0, 1],
    ]

    assert ak.to_list(ak.sort(v1_array, axis=1)) == [
        [1, 2, 3],
        [],
        None,
        [4, 5],
    ]

    assert ak.to_list(v1_array[ak.argsort(v1_array, axis=1)]) == ak.to_list(
        ak.sort(v1_array, axis=1)
    )

    v2_array = v1_to_v2(v1_array.layout)
    assert ak.to_list(v2_array.sort(axis=1)) == ak.to_list(ak.sort(v1_array, axis=1))
    assert v2_array.typetracer.sort(axis=1).form == v2_array.argsort(axis=1).form

    assert ak.to_list(v2_array.argsort(axis=1)) == ak.to_list(
        ak.argsort(v1_array, axis=1)
    )


def test_keep_None_in_place_test_2():
    v1_array = ak.Array([[3, 2, 1], [], None, [4, 5]])
    v2_array = v1_to_v2(v1_array.layout)
    assert v2_array.typetracer.argsort(axis=1).form == v2_array.argsort(axis=1).form


def test_empty_slice():
    electron = ak.Array([[], [{"pt": 1.0}]], with_name="electron")

    electron = electron[electron.pt > 5]
    v2_electron = v1_to_v2(electron.layout)
    v2_electron = v2_electron[[[], []]]

    assert ak.to_list(electron) == [[], []]
    assert ak.to_list(v2_electron) == [[], []]

    id = ak.argsort(electron, axis=1)

    assert ak.to_list(electron[id]) == [[], []]
    assert ak.to_list(v2_electron[id]) == [[], []]
    assert v2_electron.typetracer[id].form == v2_electron[id].form


def test_masked():
    v1_array = ak.Array([[0, 1, 2, 3], [3, 3, 3, 2, 1]])
    is_valid = v1_array != 3

    assert v1_array.mask[is_valid].tolist() == [
        [0, 1, 2, None],
        [None, None, None, 2, 1],
    ]

    assert ak.sort(v1_array.mask[is_valid]).tolist() == [
        [0, 1, 2, None],
        [1, 2, None, None, None],
    ]

    v2_array = v1_to_v2(v1_array.layout)
    v2_array_mask = ak._v2.contents.ListOffsetArray(
        v2_array.offsets,
        ak._v2.contents.ByteMaskedArray(
            ak._v2.index.Index8(is_valid.layout.content),
            v2_array.content,
            valid_when=True,
        ),
    )

    assert ak.to_list(v2_array_mask) == [[0, 1, 2, None], [None, None, None, 2, 1]]

    assert ak.to_list(v2_array_mask.sort(axis=1)) == [
        [0, 1, 2, None],
        [1, 2, None, None, None],
    ]
    assert v2_array_mask.typetracer.sort(axis=1).form == v2_array_mask.sort(axis=1).form


def test_v1_argsort_and_v2_sort():
    v1_array = ak.Array([1, 2, None, 3, 0, None])
    assert ak.argsort(v1_array).tolist() == [4, 0, 1, 3, 2, 5]
    assert v1_array[ak.argsort(v1_array)].tolist() == [
        0,
        1,
        2,
        3,
        None,
        None,
    ]

    v2_array = v1_to_v2(v1_array.layout)
    assert ak.to_list(v2_array.sort()) == ak.sort(v1_array).tolist()
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_v1_argsort_2d_and_v2_sort():
    v1_array = ak.Array([[1, 2, None, 3, 0, None], [1, 2, None, 3, 0, None]])
    assert ak.argsort(v1_array).tolist() == [[4, 0, 1, 3, 2, 5], [4, 0, 1, 3, 2, 5]]
    assert v1_array[ak.argsort(v1_array)].tolist() == [
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

    v2_array = v1_to_v2(v1_array.layout)
    assert ak.to_list(v2_array.sort()) == ak.sort(v1_array).tolist()
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_nan():
    v1_array = ak.Array([1, 2, np.nan, 3, 0, np.nan])

    assert ak.argsort(v1_array).tolist() == [
        2,
        5,
        4,
        0,
        1,
        3,
    ]
    # Note, `nan` comparison with `nan` returns False
    assert str(ak.sort(v1_array).tolist()) == "[nan, nan, 0.0, 1.0, 2.0, 3.0]"

    v2_array = v1_to_v2(v1_array.layout)
    assert str(ak.to_list(v2_array.sort())) == "[nan, nan, 0.0, 1.0, 2.0, 3.0]"
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_sort_strings():
    v1_array = ak.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    )
    v2_array = v1_to_v2(v1_array.layout)
    assert ak.to_list(v2_array) == [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
    ]
    assert ak.to_list(v2_array.sort()) == [
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
    v1_array = ak.Array(
        [["one", "two"], ["three", "four", "five"], ["six"], ["seven", "eight"]]
    )
    v2_array = v1_to_v2(v1_array.layout)
    assert ak.to_list(v2_array) == [
        ["one", "two"],
        ["three", "four", "five"],
        ["six"],
        ["seven", "eight"],
    ]
    assert ak.to_list(v2_array.sort()) == [
        ["one", "two"],
        ["five", "four", "three"],
        ["six"],
        ["eight", "seven"],
    ]

    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_sort_invalid_axis():
    v1_array = ak.from_numpy(
        np.array([[3.3, 2.2], [1.1, 5.5], [4.4, 6.6]]),
        regulararray=True,
        highlevel=False,
    )
    v2_array = v1_to_v2(v1_array)

    with pytest.raises(ValueError) as err:
        v2_array.sort(axis=3)
    assert str(err.value).startswith(
        "axis=3 exceeds the depth of the nested list structure (which is 2)"
    )


def test_numpy_array_iscontiguous():
    matrix = np.arange(64).reshape(8, -1)
    layout = ak.layout.NumpyArray(matrix[:, 0])
    assert ak.to_list(layout) == [0, 8, 16, 24, 32, 40, 48, 56]

    assert not layout.iscontiguous
    v2_layout = v1_to_v2(layout)
    assert not v2_layout.is_contiguous

    matrix2 = np.arange(64).reshape(8, -1)
    v2_array = ak._v2.contents.NumpyArray(matrix2[:, 0])
    assert not v2_array.is_contiguous

    assert ak.to_list(v2_array.sort()) == [0, 8, 16, 24, 32, 40, 48, 56]
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_numpyarray_sort():
    v1_array = ak.from_numpy(
        np.array([3.3, 2.2, 1.1, 5.5, 4.4]), regulararray=True, highlevel=False
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(np.sort(v2_array)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert ak.to_list(v2_array.sort()) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form


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
    assert ak.to_list(
        ak.argsort(array, axis=2, ascending=True, stable=False)
    ) == ak.to_list(np.argsort(array, 2))
    assert ak.to_list(
        ak.sort(array, axis=2, ascending=True, stable=False)
    ) == ak.to_list(np.sort(array, 2))
    assert ak.to_list(
        ak.argsort(array, axis=1, ascending=True, stable=False)
    ) == ak.to_list(np.argsort(array, 1))
    assert ak.to_list(
        ak.sort(array, axis=1, ascending=True, stable=False)
    ) == ak.to_list(np.sort(array, 1))
    assert ak.to_list(ak.sort(array, axis=1, ascending=False, stable=False)) == [
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
    assert ak.to_list(
        ak.sort(array, axis=0, ascending=True, stable=False)
    ) == ak.to_list(np.sort(array, 0))
    assert ak.to_list(
        ak.argsort(array, axis=0, ascending=True, stable=False)
    ) == ak.to_list(np.argsort(array, 0))


def test_bool_sort():
    v1_array = ak.from_numpy(
        np.array([True, False, True, False, False]), regulararray=True, highlevel=False
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v2_array.sort()) == [
        False,
        False,
        False,
        True,
        True,
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_emptyarray_sort():
    v2_array = ak._v2.contents.emptyarray.EmptyArray()
    assert ak.to_list(v2_array.sort()) == []

    v1_array = ak.Array([[], [], []])
    v2_array = v1_to_v2(v1_array.layout)
    assert ak.to_list(v2_array.sort()) == [[], [], []]
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_listarray_sort():
    v2_array = ak._v2.contents.listarray.ListArray(  # noqa: F841
        ak._v2.index.Index(np.array([4, 100, 1])),
        ak._v2.index.Index(np.array([7, 100, 3, 200])),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array([6.6, 4.4, 5.5, 7.7, 3.3, 2.2, 1.1, 8.8])
        ),
    )

    assert ak.to_list(v2_array) == [
        [3.3, 2.2, 1.1],
        [],
        [4.4, 5.5],
    ]

    assert ak.to_list(v2_array.sort()) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_listoffsetarray_sort():
    v1_array = ak.from_iter(
        [[3.3, 2.2, 1.1], [], [5.5, 4.4], [6.6], [9.9, 7.7, 8.8, 10.1]], highlevel=False
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v2_array.sort()) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9, 10.10],
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form
    assert ak.to_list(v2_array.sort(axis=0)) == [
        [3.3, 2.2, 1.1],
        [],
        [5.5, 4.4],
        [6.6],
        [9.9, 7.7, 8.8, 10.1],
    ]
    assert v2_array.typetracer.sort(axis=0).form == v2_array.sort(axis=0).form

    v1_array = ak.from_iter(
        [
            [[11.1, 0.0, -2.2], [], [33.33, 4.4]],
            [],
            [[5.5]],
            [[6.6, -9.9, 8.8, 7.7]],
            [[], [12.2, 1.1, 10.0]],
        ],
        highlevel=False,
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v2_array.sort(axis=0)) == ak.to_list(ak.sort(v1_array, axis=0))
    assert v2_array.typetracer.sort(axis=0).form == v2_array.sort(axis=0).form
    # [
    #     [[5.5, -9.9, -2.2], [], [33.33, 4.4]],
    #     [],
    #     [[6.6]],
    #     [[11.1, 0.0, 8.8, 7.7]],
    #     [[], [12.2, 1.1, 10.0]]
    # ]
    assert ak.to_list(v2_array.sort(axis=1)) == ak.to_list(ak.sort(v1_array, axis=1))
    assert v2_array.typetracer.sort(axis=1).form == v2_array.sort(axis=1).form
    # [
    #     [[11.1, 0.0, -2.2], [], [33.33, 4.4]],
    #     [],
    #     [[5.5]],
    #     [[6.6, -9.9, 8.8, 7.7]],
    #     [[], [12.2, 1.1, 10.0]]
    # ]
    assert ak.to_list(v2_array.sort(axis=2)) == ak.to_list(ak.sort(v1_array, axis=2))
    assert v2_array.typetracer.sort(axis=2).form == v2_array.sort(axis=2).form
    # [
    #     [[-2.2, 0.0, 11.1], [], [4.4, 33.33]],
    #     [],
    #     [[5.5]],
    #     [[-9.9, 6.6, 7.7, 8.8]],
    #     [[], [1.1, 10.0, 12.2]]
    # ]
    assert ak.to_list(v2_array.sort()) == ak.to_list(ak.sort(v1_array))
    assert v2_array.typetracer.sort().form == v2_array.sort().form
    # [
    #     [[-2.2, 0.0, 11.1], [], [4.4, 33.33]],
    #     [],
    #     [[5.5]],
    #     [[-9.9, 6.6, 7.7, 8.8]],
    #     [[], [1.1, 10.0, 12.2]]
    # ]


def test_regulararray_sort():
    v1_array = ak.from_numpy(
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
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v2_array) == [
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
    assert ak.to_list(v2_array.sort()) == [
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
    content = ak.from_iter(
        [
            [[1.1, 0.0, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 9.9, 8.8, 7.7]],
            [[], [12.2, 11.1, 10.0]],
        ],
        highlevel=False,
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    v1_array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    v2_array = v1_to_v2(v1_array)

    assert ak.to_list(v1_array) == [
        [[1.1, 0.0, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [12.2, 11.1, 10.0]],
    ]
    assert ak.to_list(ak.sort(v1_array)) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert ak.to_list(v2_array.sort()) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form


def test_bytemaskedarray_sort_2():
    array3 = ak.Array(
        [[2.2, 1.1, 3.3], [], [4.4, 5.5], [5.5], [-4.4, -5.5, -6.6]]
    ).layout

    assert ak.to_list(ak.sort(array3, axis=1, ascending=False, stable=False)) == [
        [3.3, 2.2, 1.1],
        [],
        [5.5, 4.4],
        [5.5],
        [-4.4, -5.5, -6.6],
    ]

    assert ak.to_list(ak.sort(array3, axis=0, ascending=True, stable=False)) == [
        [-4.4, -5.5, -6.6],
        [],
        [2.2, 1.1],
        [4.4],
        [5.5, 5.5, 3.3],
    ]

    content = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    assert ak.to_list(ak.argsort(array, axis=0, ascending=True, stable=False)) == [
        [0, 0, 0],
        [],
        [2, 2, 2, 2],
        None,
        None,
    ]

    assert ak.to_list(ak.sort(array, axis=0, ascending=True, stable=False)) == [
        [0.0, 1.1, 2.2],
        [],
        [6.6, 7.7, 8.8, 9.9],
        None,
        None,
    ]

    assert ak.to_list(ak.sort(array, axis=0, ascending=False, stable=False)) == [
        [6.6, 7.7, 8.8],
        [],
        [0.0, 1.1, 2.2, 9.9],
        None,
        None,
    ]

    assert ak.to_list(ak.argsort(array, axis=1, ascending=True, stable=False)) == [
        [0, 1, 2],
        [],
        None,
        None,
        [0, 1, 2, 3],
    ]

    assert ak.to_list(array.sort(1, False, False)) == [
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

    assert ak.to_list(v2_array.sort()) == [
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
    assert ak.to_list(v2_array.sort()) == [0.0, 1.1, 2.2, 3.3]
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

    assert ak.to_list(v2_array) == [5.5, 4.4, 1, 2, 3.3, 3, 5.5]
    assert ak.to_list(v2_array.sort()) == [1.0, 2.0, 3.0, 3.3, 4.4, 5.5, 5.5]


def test_unionarray_sort_2():
    v2_array = ak._v2.contents.unionarray.UnionArray(  # noqa: F841
        ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak._v2.contents.numpyarray.NumpyArray(np.array([1, 2, 3])),
            ak._v2.contents.numpyarray.NumpyArray(np.array([7, 0, 3, 4, 5])),
        ],
    )

    assert ak.to_list(v2_array) == [5, 4, 1, 2, 3, 3, 5]
    assert ak.to_list(v2_array.sort()) == [1, 2, 3, 3, 4, 5, 5]


def test_indexedarray_sort():
    v2_array = ak._v2.contents.indexedarray.IndexedArray(  # noqa: F841
        ak._v2.index.Index(np.array([2, 2, 0, 1, 4, 5, 4])),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    assert ak.to_list(v2_array) == [3.3, 3.3, 1.1, 2.2, 5.5, 6.6, 5.5]
    assert ak.to_list(v2_array.sort()) == [1.1, 2.2, 3.3, 3.3, 5.5, 5.5, 6.6]
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
    assert ak.to_list(v2_array.sort()) == [
        {"nest": 2.2},
        {"nest": 3.3},
        {"nest": 3.3},
        {"nest": 5.5},
        {"nest": 6.6},
        None,
        None,
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form

    v1_array = ak.Array([[1, 2, None, 3, 0, None], [1, 2, None, 3, 0, None]])
    v2_array = v1_to_v2(v1_array.layout)
    assert ak.to_list(v2_array) == [
        [1, 2, None, 3, 0, None],
        [1, 2, None, 3, 0, None],
    ]
    assert ak.to_list(v2_array.sort()) == [
        [0, 1, 2, 3, None, None],
        [0, 1, 2, 3, None, None],
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form

    v1_array = ak.Array(
        [
            [None, None, 2.2, 1.1, 3.3],
            [None, None, None],
            [4.4, None, 5.5],
            [5.5, None, None],
            [-4.4, -5.5, -6.6],
        ]
    )
    v2_array = v1_to_v2(v1_array.layout)

    assert ak.to_list(v2_array.sort()) == [
        [1.1, 2.2, 3.3, None, None],
        [None, None, None],
        [4.4, 5.5, None],
        [5.5, None, None],
        [-6.6, -5.5, -4.4],
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form

    assert ak.to_list(v2_array.sort(axis=0)) == [
        [-4.4, -5.5, -6.6, 1.1, 3.3],
        [4.4, None, 2.2],
        [5.5, None, 5.5],
        [None, None, None],
        [None, None, None],
    ]
    assert v2_array.typetracer.sort(axis=0).form == v2_array.sort(axis=0).form

    assert ak.to_list(v2_array.sort(axis=1, ascending=True, stable=False)) == [
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

    assert ak.to_list(v2_array.sort(axis=1, ascending=False, stable=True)) == [
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

    assert ak.to_list(v2_array.sort(axis=1, ascending=False, stable=False)) == [
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
    array = ak.layout.IndexedArray64(
        ak.layout.Index64([]), ak.layout.NumpyArray([1, 2, 3])
    )
    assert ak.to_list(array) == []
    assert ak.to_list(ak.sort(array)) == []
    assert ak.to_list(ak.argsort(array)) == []

    content0 = ak.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    content1 = ak.from_iter(["one", "two", "three", "four", "five"], highlevel=False)
    tags = ak.layout.Index8([])
    index = ak.layout.Index32([])
    array = ak.layout.UnionArray8_32(tags, index, [content0, content1])
    assert ak.to_list(array) == []
    assert ak.to_list(ak.sort(array)) == []
    assert ak.to_list(ak.argsort(array)) == []

    content = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.layout.Index8([])
    array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    assert ak.to_list(array) == []
    assert ak.to_list(ak.sort(array)) == []
    assert ak.to_list(ak.argsort(array)) == []

    array = ak.layout.NumpyArray([])
    assert ak.to_list(array) == []
    assert ak.to_list(ak.sort(array)) == []
    assert ak.to_list(ak.argsort(array)) == []

    array = ak.layout.RecordArray([])
    assert ak.to_list(array) == []
    assert ak.to_list(ak.sort(array)) == []
    assert ak.to_list(ak.argsort(array)) == []

    content = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts1 = ak.layout.Index64([])
    stops1 = ak.layout.Index64([])
    offsets1 = ak.layout.Index64(np.array([0]))
    array = ak.layout.ListArray64(starts1, stops1, content)
    assert ak.to_list(array) == []
    assert ak.to_list(ak.sort(array)) == []
    assert ak.to_list(ak.argsort(array)) == []

    array = ak.layout.ListOffsetArray64(offsets1, content)
    assert ak.to_list(array) == []
    assert ak.to_list(ak.sort(array)) == []
    assert ak.to_list(ak.argsort(array)) == []


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
    assert ak.to_list(v2_array.sort()) == [
        [{"nest": 0.0}, {"nest": 1.1}, {"nest": 2.2}],
        [{"nest": 4.4}, {"nest": 5.5}, {"nest": 33.33}],
    ]
    assert v2_array.typetracer.sort().form == v2_array.sort().form
