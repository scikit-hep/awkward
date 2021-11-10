# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_bool_sort():
    array = ak.layout.NumpyArray(np.array([True, False, True, False, False]))
    array = v1_to_v2(array)
    assert ak.to_list(array.sort(axis=0, ascending=True, stable=False)) == [
        False,
        False,
        False,
        True,
        True,
    ]


def test_keep_None_in_place_test():
    array = ak.Array([[3, 2, 1], [], None, [4, 5]])
    array = v1_to_v2(array.layout)

    assert ak.to_list(array.argsort(axis=1)) == [
        [2, 1, 0],
        [],
        None,
        [0, 1],
    ]

    assert ak.to_list(array.sort(axis=1)) == [
        [1, 2, 3],
        [],
        None,
        [4, 5],
    ]


def test_slicing_FIXME():
    # awkward/_v2/_slicing.py:218:
    array = ak.Array([[3, 2, 1], [], None, [4, 5]])
    array = v1_to_v2(array.layout)

    assert ak.to_list(array[array.argsort(axis=1)]) == ak.to_list(array.sort(axis=1))


def test_EmptyArray():
    array = ak.layout.EmptyArray()
    array = v1_to_v2(array)

    assert ak.to_list(array.sort()) == []
    assert ak.to_list(array.argsort()) == []

    array2 = ak.Array([[], [], []])
    array2 = v1_to_v2(array2.layout)

    assert ak.to_list(array2.argsort()) == [[], [], []]


def test_EmptyArray_type_FIXME():
    array = ak.layout.EmptyArray()
    array = v1_to_v2(array)

    assert str(array.sort().form.type) == "unknown"
    assert str(array.argsort().form.type) == "int64"

    array2 = ak.Array([[], [], []])
    array2 = v1_to_v2(array2.layout)
    assert str(array2.argsort().form.type) == "var * int64"


def test_NumpyArray():
    array = ak.layout.NumpyArray(np.array([3.3, 2.2, 1.1, 5.5, 4.4]))
    array = v1_to_v2(array)

    assert ak.to_list(array.argsort(axis=0, ascending=True, stable=False)) == [
        2,
        1,
        0,
        4,
        3,
    ]
    assert ak.to_list(array.argsort(axis=0, ascending=False, stable=False)) == [
        3,
        4,
        0,
        1,
        2,
    ]

    assert ak.to_list(array.sort(axis=0, ascending=True, stable=False)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert ak.to_list(array.sort(axis=0, ascending=False, stable=False)) == [
        5.5,
        4.4,
        3.3,
        2.2,
        1.1,
    ]

    array2 = ak.layout.NumpyArray(np.array([[3.3, 2.2, 4.4], [1.1, 5.5, 3.3]]))
    array2 = v1_to_v2(array2)

    assert ak.to_list(array2.sort(axis=1, ascending=True, stable=False)) == ak.to_list(
        np.sort(array2, axis=1)
    )
    assert ak.to_list(array2.sort(axis=0, ascending=True, stable=False)) == ak.to_list(
        np.sort(array2, axis=0)
    )

    assert ak.to_list(
        array2.argsort(axis=1, ascending=True, stable=False)
    ) == ak.to_list(np.argsort(array2, 1))

    assert ak.to_list(
        array2.argsort(axis=0, ascending=True, stable=False)
    ) == ak.to_list(np.argsort(array2, 0))

    with pytest.raises(ValueError) as err:
        array2.sort(axis=2, ascending=True, stable=False)
    assert str(err.value).startswith(
        "axis=2 exceeds the depth of the nested list structure (which is 2)"
    )


def test_IndexedOptionArray():
    array = ak.Array(
        [
            [None, None, 2.2, 1.1, 3.3],
            [None, None, None],
            [4.4, None, 5.5],
            [5.5, None, None],
            [-4.4, -5.5, -6.6],
        ]
    )
    array = v1_to_v2(array.layout)

    assert ak.to_list(array.sort(axis=0, ascending=True, stable=False)) == [
        [-4.4, -5.5, -6.6, 1.1, 3.3],
        [4.4, None, 2.2],
        [5.5, None, 5.5],
        [None, None, None],
        [None, None, None],
    ]

    assert ak.to_list(array.sort(axis=1, ascending=True, stable=False)) == [
        [1.1, 2.2, 3.3, None, None],
        [None, None, None],
        [4.4, 5.5, None],
        [5.5, None, None],
        [-6.6, -5.5, -4.4],
    ]

    assert ak.to_list(array.sort(axis=1, ascending=False, stable=True)) == [
        [3.3, 2.2, 1.1, None, None],
        [None, None, None],
        [5.5, 4.4, None],
        [5.5, None, None],
        [-4.4, -5.5, -6.6],
    ]

    assert ak.to_list(array.sort(axis=1, ascending=False, stable=False)) == [
        [3.3, 2.2, 1.1, None, None],
        [None, None, None],
        [5.5, 4.4, None],
        [5.5, None, None],
        [-4.4, -5.5, -6.6],
    ]

    assert ak.to_list(array.argsort(axis=0, ascending=True, stable=True)) == [
        [4, 4, 4, 0, 0],
        [2, 0, 0],
        [3, 1, 2],
        [0, 2, 1],
        [1, 3, 3],
    ]

    assert ak.to_list(array.argsort(axis=0, ascending=True, stable=False)) == [
        [4, 4, 4, 0, 0],
        [2, 0, 0],
        [3, 1, 2],
        [0, 2, 1],
        [1, 3, 3],
    ]

    assert ak.to_list(array.argsort(axis=0, ascending=False, stable=True)) == [
        [3, 4, 2, 0, 0],
        [2, 0, 0],
        [4, 1, 4],
        [0, 2, 1],
        [1, 3, 3],
    ]
    assert ak.to_list(array.argsort(axis=0, ascending=False, stable=False)) == [
        [3, 4, 2, 0, 0],
        [2, 0, 0],
        [4, 1, 4],
        [0, 2, 1],
        [1, 3, 3],
    ]

    assert ak.to_list(array.argsort(axis=1, ascending=True, stable=True)) == [
        [3, 2, 4, 0, 1],
        [0, 1, 2],
        [0, 2, 1],
        [0, 1, 2],
        [2, 1, 0],
    ]

    assert ak.to_list(array.argsort(axis=1, ascending=True, stable=False)) == [
        [3, 2, 4, 0, 1],
        [0, 1, 2],
        [0, 2, 1],
        [0, 1, 2],
        [2, 1, 0],
    ]

    assert ak.to_list(array.argsort(axis=1, ascending=False, stable=True)) == [
        [4, 2, 3, 0, 1],
        [0, 1, 2],
        [2, 0, 1],
        [0, 1, 2],
        [0, 1, 2],
    ]

    array2 = ak.Array([None, None, 1, -1, 30])
    array2 = v1_to_v2(array2.layout)

    assert ak.to_list(array2.argsort(axis=0, ascending=True, stable=True)) == [
        3,
        2,
        4,
        0,
        1,
    ]

    array3 = ak.Array(
        [[2.2, 1.1, 3.3], [], [4.4, 5.5], [5.5], [-4.4, -5.5, -6.6]]
    ).layout
    array3 = v1_to_v2(array3)

    assert ak.to_list(array3.sort(axis=1, ascending=False, stable=False)) == [
        [3.3, 2.2, 1.1],
        [],
        [5.5, 4.4],
        [5.5],
        [-4.4, -5.5, -6.6],
    ]

    assert ak.to_list(array3.sort(axis=0, ascending=True, stable=False)) == [
        [-4.4, -5.5, -6.6],
        [],
        [2.2, 1.1],
        [4.4],
        [5.5, 5.5, 3.3],
    ]


def test_IndexedArray():
    content = ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    index1 = ak.layout.Index32(np.array([1, 2, 3, 4], dtype=np.int32))
    indexedarray1 = ak.layout.IndexedArray32(index1, content)
    indexedarray1 = v1_to_v2(indexedarray1)

    assert ak.to_list(indexedarray1.argsort(axis=0, ascending=True, stable=False)) == [
        0,
        1,
        2,
        3,
    ]

    index2 = ak._v2.index.Index(np.array([1, 2, 3], dtype=np.int64))
    indexedarray2 = ak._v2.contents.IndexedArray(index2, indexedarray1)

    assert ak.to_list(indexedarray2.sort(axis=0, ascending=False, stable=False)) == [
        5.5,
        4.4,
        3.3,
    ]

    index3 = ak._v2.index.Index32(np.array([1, 2], dtype=np.int32))
    indexedarray3 = ak._v2.contents.IndexedArray(index3, indexedarray2)

    assert ak.to_list(indexedarray3.sort(axis=0, ascending=True, stable=False)) == [
        4.4,
        5.5,
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
    sorted = ak.argsort(array, axis=1, ascending=True, stable=False)
    assert ak.to_list(sorted) == ak.to_list(np.argsort(array, 1))

    array = v1_to_v2(array)

    sorted = array.argsort(axis=2, ascending=True, stable=False)
    assert ak.to_list(sorted) == ak.to_list(np.argsort(array, 2))

    sorted = array.sort(axis=2, ascending=True, stable=False)
    assert ak.to_list(sorted) == ak.to_list(np.sort(array, 2))

    sorted = array.argsort(axis=1, ascending=True, stable=False)

    assert ak.to_list(sorted) == ak.to_list(np.argsort(array, 1))

    sorted = array.sort(axis=1, ascending=True, stable=False)
    assert ak.to_list(sorted) == ak.to_list(np.sort(array, 1))

    sorted = array.sort(axis=1, ascending=False, stable=False)
    assert ak.to_list(sorted) == [
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

    sorted = array.sort(axis=0, ascending=True, stable=False)
    assert ak.to_list(sorted) == ak.to_list(np.sort(array, 0))

    assert ak.to_list(
        array.argsort(axis=0, ascending=True, stable=False)
    ) == ak.to_list(np.argsort(array, 0))


def test_ByteMaskedArray():
    content = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    array = v1_to_v2(array)
    sorted = array.argsort(axis=0, ascending=True, stable=False)
    assert ak.to_list(sorted) == [
        [0, 0, 0],
        [],
        [2, 2, 2, 2],
        None,
        None,
    ]

    sorted = array.sort(axis=0, ascending=True, stable=False)
    assert ak.to_list(sorted) == [
        [0.0, 1.1, 2.2],
        [],
        [6.6, 7.7, 8.8, 9.9],
        None,
        None,
    ]

    assert ak.to_list(array.sort(axis=0, ascending=False, stable=False)) == [
        [6.6, 7.7, 8.8],
        [],
        [0.0, 1.1, 2.2, 9.9],
        None,
        None,
    ]

    assert ak.to_list(array.argsort(axis=1, ascending=True, stable=False)) == [
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


def test_UnionArray():
    content0 = ak.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    content1 = ak.from_iter(
        [["one"], ["two"], ["three"], ["four"], ["five"]], highlevel=False
    )
    tags = ak.layout.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))
    index = ak.layout.Index32(np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    array = ak.layout.UnionArray8_32(tags, index, [content0, content1])
    array = v1_to_v2(array)

    with pytest.raises(ValueError):
        array.sort(axis=1, ascending=True, stable=False)


def test_sort_strings():
    content = ak.from_iter(["one", "two", "three", "four", "five"], highlevel=False)
    assert ak.to_list(content) == ["one", "two", "three", "four", "five"]
    content = v1_to_v2(content)

    assert ak.to_list(content.sort(axis=0, ascending=True, stable=False)) == [
        "five",
        "four",
        "one",
        "three",
        "two",
    ]
    assert ak.to_list(content.sort(axis=0, ascending=False, stable=False)) == [
        "two",
        "three",
        "one",
        "four",
        "five",
    ]


def test_sort_bytestrings():
    array = ak.from_iter(
        [b"one", b"two", b"three", b"two", b"two", b"one", b"three"], highlevel=False
    )
    array = v1_to_v2(array)
    assert ak.to_list(array) == [
        b"one",
        b"two",
        b"three",
        b"two",
        b"two",
        b"one",
        b"three",
    ]

    assert ak.to_list(array.sort(axis=0, ascending=True, stable=False)) == [
        b"one",
        b"one",
        b"three",
        b"three",
        b"two",
        b"two",
        b"two",
    ]

    assert ak.to_list(array.argsort(axis=0, ascending=True, stable=True)) == [
        0,
        5,
        2,
        6,
        1,
        3,
        4,
    ]


def test_sort_zero_length_arrays():
    array = ak.layout.IndexedArray64(
        ak.layout.Index64([]), ak.layout.NumpyArray([1, 2, 3])
    )
    array = v1_to_v2(array)
    assert ak.to_list(array) == []
    assert ak.to_list(array.sort()) == []
    assert ak.to_list(array.argsort()) == []

    content = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.layout.Index8([])
    array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    array = v1_to_v2(array)
    assert ak.to_list(array) == []
    assert ak.to_list(array.sort()) == []
    assert ak.to_list(array.argsort()) == []

    array = ak.layout.NumpyArray([])
    array = v1_to_v2(array)
    assert ak.to_list(array) == []
    assert ak.to_list(array.sort()) == []
    assert ak.to_list(array.argsort()) == []

    array = ak.layout.RecordArray([])
    array = v1_to_v2(array)
    assert ak.to_list(array) == []
    assert ak.to_list(array.sort()) == []
    assert ak.to_list(array.argsort()) == []

    content = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts1 = ak.layout.Index64([])
    stops1 = ak.layout.Index64([])
    offsets1 = ak.layout.Index64(np.array([0]))
    array = ak.layout.ListArray64(starts1, stops1, content)
    array = v1_to_v2(array)
    assert ak.to_list(array) == []
    assert ak.to_list(array.sort()) == []
    assert ak.to_list(array.argsort()) == []

    array = ak.layout.ListOffsetArray64(offsets1, content)
    array = v1_to_v2(array)
    assert ak.to_list(array) == []
    assert ak.to_list(array.sort()) == []
    assert ak.to_list(array.argsort()) == []


def test_UnionArray_FIXME():
    content0 = ak.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    content1 = ak.from_iter(["one", "two", "three", "four", "five"], highlevel=False)
    tags = ak.layout.Index8([])
    index = ak.layout.Index32([])
    array = ak.layout.UnionArray8_32(tags, index, [content0, content1])
    array = v1_to_v2(array)
    assert ak.to_list(array) == []

    assert ak.to_list(array.sort()) == []
    assert ak.to_list(array.argsort()) == []
