# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_bool_sort():
    array = ak.contents.NumpyArray(np.array([True, False, True, False, False]))
    assert to_list(ak.operations.sort(array, axis=0, ascending=True, stable=False)) == [
        False,
        False,
        False,
        True,
        True,
    ]


def test_keep_None_in_place_test():
    array = ak.highlevel.Array([[3, 2, 1], [], None, [4, 5]]).layout

    assert to_list(ak.operations.argsort(array, axis=1)) == [
        [2, 1, 0],
        [],
        None,
        [0, 1],
    ]

    assert to_list(ak.operations.sort(array, axis=1)) == [
        [1, 2, 3],
        [],
        None,
        [4, 5],
    ]


def test_slicing_FIXME():
    # awkward/_slicing.py:218:
    array = ak.highlevel.Array([[3, 2, 1], [], None, [4, 5]]).layout

    assert to_list(array[ak.operations.argsort(array, axis=1)]) == to_list(
        ak.operations.sort(array, axis=1)
    )


def test_EmptyArray():
    array = ak.contents.EmptyArray()

    assert to_list(ak.operations.sort(array)) == []
    assert (
        to_list(
            ak.operations.argsort(
                array,
            )
        )
        == []
    )

    array2 = ak.highlevel.Array([[], [], []]).layout

    assert to_list(ak.operations.argsort(array2)) == [[], [], []]


def test_EmptyArray_type():
    array = ak.contents.EmptyArray()

    assert str(ak.operations.sort(array).layout.form.type) == "unknown"
    assert str(ak.operations.type(ak.operations.argsort(array))) == "0 * int64"

    array2 = ak.highlevel.Array([[], [], []]).layout
    assert str(ak.operations.type(ak.operations.argsort(array2))) == "3 * var * int64"


def test_NumpyArray():
    array = ak.contents.NumpyArray(np.array([3.3, 2.2, 1.1, 5.5, 4.4]))

    assert to_list(
        ak.operations.argsort(array, axis=0, ascending=True, stable=False)
    ) == [
        2,
        1,
        0,
        4,
        3,
    ]
    assert to_list(
        ak.operations.argsort(array, axis=0, ascending=False, stable=False)
    ) == [
        3,
        4,
        0,
        1,
        2,
    ]

    assert to_list(ak.operations.sort(array, axis=0, ascending=True, stable=False)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert to_list(
        ak.operations.sort(array, axis=0, ascending=False, stable=False)
    ) == [
        5.5,
        4.4,
        3.3,
        2.2,
        1.1,
    ]

    array2 = ak.contents.NumpyArray(np.array([[3.3, 2.2, 4.4], [1.1, 5.5, 3.3]]))

    assert to_list(
        ak.operations.sort(array2, axis=1, ascending=True, stable=False)
    ) == to_list(np.sort(np.asarray(array2), axis=1))
    assert to_list(
        ak.operations.sort(array2, axis=0, ascending=True, stable=False)
    ) == to_list(np.sort(np.asarray(array2), axis=0))

    assert to_list(
        ak.operations.argsort(array2, axis=1, ascending=True, stable=False)
    ) == to_list(np.argsort(np.asarray(array2), 1))

    assert to_list(
        ak.operations.argsort(array2, axis=0, ascending=True, stable=False)
    ) == to_list(np.argsort(np.asarray(array2), 0))

    with pytest.raises(ValueError) as err:
        ak.operations.argsort(array2, axis=2, ascending=True, stable=False)
    assert "axis=2 exceeds the depth of the nested list structure (which is 2)" in str(
        err.value
    )


def test_IndexedOptionArray():
    array = ak.highlevel.Array(
        [
            [None, None, 2.2, 1.1, 3.3],
            [None, None, None],
            [4.4, None, 5.5],
            [5.5, None, None],
            [-4.4, -5.5, -6.6],
        ]
    ).layout

    assert to_list(ak.operations.sort(array, axis=0, ascending=True, stable=False)) == [
        [-4.4, -5.5, -6.6, 1.1, 3.3],
        [4.4, None, 2.2],
        [5.5, None, 5.5],
        [None, None, None],
        [None, None, None],
    ]

    assert to_list(ak.operations.sort(array, axis=1, ascending=True, stable=False)) == [
        [1.1, 2.2, 3.3, None, None],
        [None, None, None],
        [4.4, 5.5, None],
        [5.5, None, None],
        [-6.6, -5.5, -4.4],
    ]

    assert to_list(ak.operations.sort(array, axis=1, ascending=False, stable=True)) == [
        [3.3, 2.2, 1.1, None, None],
        [None, None, None],
        [5.5, 4.4, None],
        [5.5, None, None],
        [-4.4, -5.5, -6.6],
    ]

    assert to_list(
        ak.operations.sort(array, axis=1, ascending=False, stable=False)
    ) == [
        [3.3, 2.2, 1.1, None, None],
        [None, None, None],
        [5.5, 4.4, None],
        [5.5, None, None],
        [-4.4, -5.5, -6.6],
    ]

    assert to_list(
        ak.operations.argsort(array, axis=0, ascending=True, stable=True)
    ) == [
        [4, 4, 4, 0, 0],
        [2, 0, 0],
        [3, 1, 2],
        [0, 2, 1],
        [1, 3, 3],
    ]

    assert to_list(
        ak.operations.argsort(array, axis=0, ascending=True, stable=False)
    ) == [
        [4, 4, 4, 0, 0],
        [2, 0, 0],
        [3, 1, 2],
        [0, 2, 1],
        [1, 3, 3],
    ]

    assert to_list(
        ak.operations.argsort(array, axis=0, ascending=False, stable=True)
    ) == [
        [3, 4, 2, 0, 0],
        [2, 0, 0],
        [4, 1, 4],
        [0, 2, 1],
        [1, 3, 3],
    ]
    assert to_list(
        ak.operations.argsort(array, axis=0, ascending=False, stable=False)
    ) == [
        [3, 4, 2, 0, 0],
        [2, 0, 0],
        [4, 1, 4],
        [0, 2, 1],
        [1, 3, 3],
    ]

    assert to_list(
        ak.operations.argsort(array, axis=1, ascending=True, stable=True)
    ) == [
        [3, 2, 4, 0, 1],
        [0, 1, 2],
        [0, 2, 1],
        [0, 1, 2],
        [2, 1, 0],
    ]

    assert to_list(
        ak.operations.argsort(array, axis=1, ascending=True, stable=False)
    ) == [
        [3, 2, 4, 0, 1],
        [0, 1, 2],
        [0, 2, 1],
        [0, 1, 2],
        [2, 1, 0],
    ]

    assert to_list(
        ak.operations.argsort(array, axis=1, ascending=False, stable=True)
    ) == [
        [4, 2, 3, 0, 1],
        [0, 1, 2],
        [2, 0, 1],
        [0, 1, 2],
        [0, 1, 2],
    ]

    array2 = ak.highlevel.Array([None, None, 1, -1, 30]).layout

    assert to_list(
        ak.operations.argsort(array2, axis=0, ascending=True, stable=True)
    ) == [
        3,
        2,
        4,
        0,
        1,
    ]

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


def test_IndexedArray():
    content = ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    index1 = ak.index.Index32(np.array([1, 2, 3, 4], dtype=np.int32))
    indexedarray1 = ak.contents.IndexedArray(index1, content)

    assert to_list(
        ak.operations.argsort(indexedarray1, axis=0, ascending=True, stable=False)
    ) == [
        0,
        1,
        2,
        3,
    ]

    index2 = ak.index.Index(np.array([1, 2, 3], dtype=np.int64))
    indexedarray2 = ak.contents.IndexedArray.simplified(index2, indexedarray1)

    assert to_list(
        ak.operations.sort(indexedarray2, axis=0, ascending=False, stable=False)
    ) == [
        5.5,
        4.4,
        3.3,
    ]

    index3 = ak.index.Index32(np.array([1, 2], dtype=np.int32))
    indexedarray3 = ak.contents.IndexedArray.simplified(index3, indexedarray2)

    assert to_list(
        ak.operations.sort(indexedarray3, axis=0, ascending=True, stable=False)
    ) == [
        4.4,
        5.5,
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

    sorted = ak.operations.argsort(array, axis=1, ascending=True, stable=False)
    assert to_list(sorted) == to_list(np.argsort(np.asarray(array), 1))

    sorted = ak.operations.argsort(array, axis=2, ascending=True, stable=False)
    assert to_list(sorted) == to_list(np.argsort(np.asarray(array), 2))

    sorted = ak.operations.sort(array, axis=2, ascending=True, stable=False)
    assert to_list(sorted) == to_list(np.sort(np.asarray(array), 2))

    sorted = ak.operations.argsort(array, axis=1, ascending=True, stable=False)

    assert to_list(sorted) == to_list(np.argsort(np.asarray(array), 1))

    sorted = ak.operations.sort(array, axis=1, ascending=True, stable=False)
    assert to_list(sorted) == to_list(np.sort(np.asarray(array), 1))

    sorted = ak.operations.sort(array, axis=1, ascending=False, stable=False)
    assert to_list(sorted) == [
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

    sorted = ak.operations.sort(array, axis=0, ascending=True, stable=False)
    assert to_list(sorted) == to_list(np.sort(np.asarray(array), 0))

    assert to_list(
        ak.operations.argsort(array, axis=0, ascending=True, stable=False)
    ) == to_list(np.argsort(np.asarray(array), 0))


def test_ByteMaskedArray():
    content = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.index.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.contents.ByteMaskedArray(mask, content, valid_when=False)
    sorted = ak.operations.argsort(array, axis=0, ascending=True, stable=False)
    assert to_list(sorted) == [
        [0, 0, 0],
        [],
        [2, 2, 2, 2],
        None,
        None,
    ]

    sorted = ak.operations.sort(array, axis=0, ascending=True, stable=False)
    assert to_list(sorted) == [
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


def test_UnionArray():
    content0 = ak.operations.from_iter(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False
    )
    content1 = ak.operations.from_iter(
        [["one"], ["two"], ["three"], ["four"], ["five"]], highlevel=False
    )
    tags = ak.index.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))
    index = ak.index.Index32(np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    array = ak.contents.UnionArray(tags, index, [content0, content1])

    with pytest.raises(ValueError):
        ak.operations.sort(array, axis=1, ascending=True, stable=False)


def test_sort_strings():
    content = ak.operations.from_iter(
        ["one", "two", "three", "four", "five"], highlevel=False
    )

    assert to_list(
        ak.operations.sort(content, axis=0, ascending=True, stable=False)
    ) == [
        "five",
        "four",
        "one",
        "three",
        "two",
    ]
    assert to_list(
        ak.operations.sort(content, axis=0, ascending=False, stable=False)
    ) == [
        "two",
        "three",
        "one",
        "four",
        "five",
    ]


def test_sort_bytestrings():
    array = ak.operations.from_iter(
        [b"one", b"two", b"three", b"two", b"two", b"one", b"three"], highlevel=False
    )
    assert to_list(array) == [
        b"one",
        b"two",
        b"three",
        b"two",
        b"two",
        b"one",
        b"three",
    ]

    assert to_list(ak.operations.sort(array, axis=0, ascending=True, stable=False)) == [
        b"one",
        b"one",
        b"three",
        b"three",
        b"two",
        b"two",
        b"two",
    ]

    assert to_list(
        ak.operations.argsort(array, axis=0, ascending=True, stable=True)
    ) == [
        0,
        5,
        2,
        6,
        1,
        3,
        4,
    ]


def test_sort_zero_length_arrays():
    array = ak.contents.IndexedArray(
        ak.index.Index64([]), ak.contents.NumpyArray([1, 2, 3])
    )
    assert to_list(array) == []
    assert to_list(ak.operations.sort(array)) == []
    assert (
        to_list(
            ak.operations.argsort(
                array,
            )
        )
        == []
    )

    content = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.index.Index8([])
    array = ak.contents.ByteMaskedArray(mask, content, valid_when=False)
    assert to_list(array) == []
    assert to_list(ak.operations.sort(array)) == []
    assert (
        to_list(
            ak.operations.argsort(
                array,
            )
        )
        == []
    )

    array = ak.contents.NumpyArray([])
    assert to_list(array) == []
    assert to_list(ak.operations.sort(array)) == []
    assert (
        to_list(
            ak.operations.argsort(
                array,
            )
        )
        == []
    )

    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts1 = ak.index.Index64([])
    stops1 = ak.index.Index64([])
    offsets1 = ak.index.Index64(np.array([0]))
    array = ak.contents.ListArray(starts1, stops1, content)
    assert to_list(array) == []
    assert to_list(ak.operations.sort(array)) == []
    assert (
        to_list(
            ak.operations.argsort(
                array,
            )
        )
        == []
    )

    array = ak.contents.ListOffsetArray(offsets1, content)
    assert to_list(array) == []
    assert to_list(ak.operations.sort(array)) == []
    assert (
        to_list(
            ak.operations.argsort(
                array,
            )
        )
        == []
    )


def test_UnionArray_FIXME():
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
    assert (
        to_list(
            ak.operations.argsort(
                array,
            )
        )
        == []
    )
