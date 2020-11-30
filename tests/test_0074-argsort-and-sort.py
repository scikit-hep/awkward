# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak


def test_keep_None_in_place_test():
    assert ak.to_list(ak.argsort(ak.Array([[3, 2, 1], [], None, [4, 5]]), axis=1)) == [
        [2, 1, 0],
        [],
        None,
        [0, 1],
    ]
    assert ak.to_list(ak.sort(ak.Array([[3, 2, 1], [], None, [4, 5]]), axis=1)) == [
        [1, 2, 3],
        [],
        None,
        [4, 5],
    ]


def test_EmptyArray():
    array = ak.layout.EmptyArray()
    assert ak.to_list(array.sort(0, True, False)) == []
    assert ak.to_list(array.argsort(0, True, False)) == []


def test_NumpyArray():
    array = ak.layout.NumpyArray(np.array([3.3, 2.2, 1.1, 5.5, 4.4]))
    assert ak.to_list(array.argsort(0, True, False)) == [2, 1, 0, 4, 3]
    assert ak.to_list(array.argsort(0, False, False)) == [3, 4, 0, 1, 2]

    assert ak.to_list(array.sort(0, True, False)) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert ak.to_list(array.sort(0, False, False)) == [5.5, 4.4, 3.3, 2.2, 1.1]

    array2 = ak.layout.NumpyArray(np.array([[3.3, 2.2, 4.4], [1.1, 5.5, 3.3]]))

    assert ak.to_list(array2.sort(1, True, False)) == ak.to_list(
        np.sort(array2, axis=1)
    )
    assert ak.to_list(array2.sort(0, True, False)) == ak.to_list(
        np.sort(array2, axis=0)
    )

    assert ak.to_list(array2.argsort(1, True, False)) == ak.to_list(
        np.argsort(array2, 1)
    )
    assert ak.to_list(array2.argsort(0, True, False)) == ak.to_list(
        np.argsort(array2, 0)
    )

    with pytest.raises(ValueError) as err:
        array2.sort(2, True, False)
    assert str(err.value).startswith(
        "axis=2 exceeds the depth of the nested list structure (which is 2)"
    )


def test_IndexedOffsetArray():
    array = ak.Array(
        [
            [2.2, 1.1, 3.3],
            [None, None, None],
            [4.4, None, 5.5],
            [5.5, None, None],
            [-4.4, -5.5, -6.6],
        ]
    ).layout

    assert ak.to_list(array.sort(0, True, False)) == [
        [-4.4, -5.5, -6.6],
        [2.2, 1.1, 3.3],
        [4.4, None, 5.5],
        [5.5, None, None],
        [None, None, None],
    ]

    assert ak.to_list(array.argsort(0, True, False)) == [
        [3, 1, 2],
        [0, 0, 0],
        [1, None, 1],
        [2, None, None],
        [None, None, None],
    ]

    assert ak.to_list(array.sort(1, True, False)) == [
        [1.1, 2.2, 3.3],
        [None, None, None],
        [4.4, 5.5, None],
        [5.5, None, None],
        [-6.6, -5.5, -4.4],
    ]

    assert ak.to_list(array.argsort(1, True, False)) == [
        [1, 0, 2],
        [None, None, None],
        [0, 1, None],
        [0, None, None],
        [2, 1, 0],
    ]

    assert ak.to_list(array.sort(1, False, False)) == [
        [3.3, 2.2, 1.1],
        [None, None, None],
        [5.5, 4.4, None],
        [5.5, None, None],
        [-4.4, -5.5, -6.6],
    ]

    assert ak.to_list(array.argsort(1, False, False)) == [
        [2, 0, 1],
        [None, None, None],
        [1, 0, None],
        [0, None, None],
        [0, 1, 2],
    ]

    array3 = ak.Array(
        [[2.2, 1.1, 3.3], [], [4.4, 5.5], [5.5], [-4.4, -5.5, -6.6]]
    ).layout

    assert ak.to_list(array3.sort(1, False, False)) == [
        [3.3, 2.2, 1.1],
        [],
        [5.5, 4.4],
        [5.5],
        [-4.4, -5.5, -6.6],
    ]

    assert ak.to_list(array3.sort(0, True, False)) == [
        [-4.4, -5.5, -6.6],
        [],
        [2.2, 1.1],
        [4.4],
        [5.5, 5.5, 3.3],
    ]

    # FIXME: Based on Numpy list sorting:
    #
    # array([list([2.2, 1.1, 3.3]), list([]), list([4.4, 5.5]), list([5.5]),
    #        list([-4.4, -5.5, -6.6])], dtype=object)
    # np.sort(array, axis=0)
    # array([list([]), list([-4.4, -5.5, -6.6]), list([2.2, 1.1, 3.3]),
    #        list([4.4, 5.5]), list([5.5])], dtype=object)
    #
    # the result should be:
    #
    # [[ -4.4, -5.5, -6.6 ],
    #  [  2.2,  1.1,  3.3 ],
    #  [  4.4,  5.5 ],
    #  [  5.5 ],
    #  []]

    # This can be done following the steps: pad, sort,
    # and dropna to strip off the None's
    #
    array4 = array3.rpad(3, 1)
    assert ak.to_list(array4) == [
        [2.2, 1.1, 3.3],
        [None, None, None],
        [4.4, 5.5, None],
        [5.5, None, None],
        [-4.4, -5.5, -6.6],
    ]

    array5 = array4.sort(0, True, False)
    assert ak.to_list(array5) == [
        [-4.4, -5.5, -6.6],
        [2.2, 1.1, 3.3],
        [4.4, 5.5, None],
        [5.5, None, None],
        [None, None, None],
    ]

    array4 = array3.rpad(5, 1)
    assert ak.to_list(array4) == [
        [2.2, 1.1, 3.3, None, None],
        [None, None, None, None, None],
        [4.4, 5.5, None, None, None],
        [5.5, None, None, None, None],
        [-4.4, -5.5, -6.6, None, None],
    ]

    array5 = array4.sort(0, True, False)
    assert ak.to_list(array5) == [
        [-4.4, -5.5, -6.6, None, None],
        [2.2, 1.1, 3.3, None, None],
        [4.4, 5.5, None, None, None],
        [5.5, None, None, None, None],
        [None, None, None, None, None],
    ]

    array5 = array4.argsort(0, True, False)
    assert ak.to_list(array5) == [
        [3, 2, 1, None, None],
        [0, 0, 0, None, None],
        [1, 1, None, None, None],
        [2, None, None, None, None],
        [None, None, None, None, None],
    ]

    # FIXME: implement dropna to strip off the None's
    #
    # array6 = array5.dropna(0)
    # assert ak.to_list(array6) == [
    #     [ -4.4, -5.5, -6.6 ],
    #     [  2.2,  1.1,  3.3 ],
    #     [  4.4,  5.5 ],
    #     [  5.5 ],
    #     []]

    assert ak.to_list(array.argsort(1, True, False)) == [
        [1, 0, 2],
        [None, None, None],
        [0, 1, None],
        [0, None, None],
        [2, 1, 0],
    ]

    assert ak.to_list(array.argsort(0, True, False)) == [
        [3, 1, 2],
        [0, 0, 0],
        [1, None, 1],
        [2, None, None],
        [None, None, None],
    ]

    content = ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    index1 = ak.layout.Index32(np.array([1, 2, 3, 4], dtype=np.int32))
    indexedarray1 = ak.layout.IndexedArray32(index1, content)
    assert ak.to_list(indexedarray1.argsort(0, True, False)) == [0, 1, 2, 3]

    index2 = ak.layout.Index64(np.array([1, 2, 3], dtype=np.int64))
    indexedarray2 = ak.layout.IndexedArray64(index2, indexedarray1)
    assert ak.to_list(indexedarray2.sort(0, False, False)) == [5.5, 4.4, 3.3]

    index3 = ak.layout.Index32(np.array([1, 2], dtype=np.int32))
    indexedarray3 = ak.layout.IndexedArray32(index3, indexedarray2)
    assert ak.to_list(indexedarray3.sort(0, True, False)) == [4.4, 5.5]


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
    assert ak.to_list(array.argsort(2, True, False)) == ak.to_list(np.argsort(array, 2))
    assert ak.to_list(array.sort(2, True, False)) == ak.to_list(np.sort(array, 2))
    assert ak.to_list(array.argsort(1, True, False)) == ak.to_list(np.argsort(array, 1))
    assert ak.to_list(array.sort(1, True, False)) == ak.to_list(np.sort(array, 1))

    assert ak.to_list(array.sort(1, False, False)) == [
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

    assert ak.to_list(array.sort(0, True, False)) == ak.to_list(np.sort(array, 0))
    assert ak.to_list(array.argsort(0, True, False)) == ak.to_list(np.argsort(array, 0))


def test_RecordArray():
    array = ak.Array(
        [
            {"x": 0.0, "y": []},
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 3, 3]},
            {"x": 4.4, "y": [4, 4, 4, 4]},
            {"x": 5.5, "y": [5, 5, 5]},
            {"x": 6.6, "y": [6, 6]},
            {"x": 7.7, "y": [7]},
            {"x": 8.8, "y": []},
        ]
    )
    assert ak.to_list(array) == [
        {"x": 0.0, "y": []},
        {"x": 1.1, "y": [1]},
        {"x": 2.2, "y": [2, 2]},
        {"x": 3.3, "y": [3, 3, 3]},
        {"x": 4.4, "y": [4, 4, 4, 4]},
        {"x": 5.5, "y": [5, 5, 5]},
        {"x": 6.6, "y": [6, 6]},
        {"x": 7.7, "y": [7]},
        {"x": 8.8, "y": []},
    ]

    assert ak.to_list(array.layout.sort(-1, True, False)) == {
        "x": [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8],
        "y": [[], [1], [2, 2], [3, 3, 3], [4, 4, 4, 4], [5, 5, 5], [6, 6], [7], []],
    }

    assert ak.to_list(array.layout.sort(-1, False, False)) == {
        "x": [8.8, 7.7, 6.6, 5.5, 4.4, 3.3, 2.2, 1.1, 0.0],
        "y": [[], [1], [2, 2], [3, 3, 3], [4, 4, 4, 4], [5, 5, 5], [6, 6], [7], []],
    }

    assert ak.to_list(array.layout.argsort(-1, True, False)) == {
        "x": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "y": [[], [0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2], [0, 1], [0], []],
    }

    assert ak.to_list(array.x.layout.argsort(0, True, False)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    ]
    assert ak.to_list(array.x.layout.argsort(0, False, False)) == [
        8,
        7,
        6,
        5,
        4,
        3,
        2,
        1,
        0,
    ]

    array_y = array.y
    assert ak.to_list(array_y) == [
        [],
        [1],
        [2, 2],
        [3, 3, 3],
        [4, 4, 4, 4],
        [5, 5, 5],
        [6, 6],
        [7],
        [],
    ]
    assert ak.to_list(array.y.layout.argsort(0, True, False)) == [
        [],
        [0],
        [1, 0],
        [2, 1, 0],
        [3, 2, 1, 0],
        [4, 3, 2],
        [5, 4],
        [6],
        [],
    ]

    assert ak.to_list(array.y.layout.argsort(1, True, True)) == [
        [],
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 2],
        [0, 1],
        [0],
        [],
    ]


def test_ByteMaskedArray():
    content = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    assert ak.to_list(array.argsort(0, True, False)) == [[0, 0, 0], [], [1, 1, 1, 0]]

    assert ak.to_list(array.sort(0, True, False)) == [
        [0.0, 1.1, 2.2],
        [],
        [6.6, 7.7, 8.8, 9.9],
    ]

    assert ak.to_list(array.sort(0, False, False)) == [
        [6.6, 7.7, 8.8],
        [],
        [0.0, 1.1, 2.2, 9.9],
    ]

    assert ak.to_list(array.argsort(1, True, False)) == [
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
    content1 = ak.from_iter(["one", "two", "three", "four", "five"], highlevel=False)
    tags = ak.layout.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))
    index = ak.layout.Index32(np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    array = ak.layout.UnionArray8_32(tags, index, [content0, content1])

    with pytest.raises(ValueError) as err:
        array.sort(1, True, False)
    assert str(err.value).startswith("cannot sort UnionArray8_32")


def test_sort_strings():
    content1 = ak.from_iter(["one", "two", "three", "four", "five"], highlevel=False)
    assert ak.to_list(content1) == ["one", "two", "three", "four", "five"]

    assert ak.to_list(content1.sort(0, True, False)) == [
        "five",
        "four",
        "one",
        "three",
        "two",
    ]
    assert ak.to_list(content1.sort(0, False, False)) == [
        "two",
        "three",
        "one",
        "four",
        "five",
    ]

    assert ak.to_list(content1.sort(1, True, False)) == [
        "five",
        "four",
        "one",
        "three",
        "two",
    ]
    assert ak.to_list(content1.sort(1, False, False)) == [
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
    assert ak.to_list(array) == [
        b"one",
        b"two",
        b"three",
        b"two",
        b"two",
        b"one",
        b"three",
    ]

    assert ak.to_list(array.sort(0, True, False)) == [
        b"one",
        b"one",
        b"three",
        b"three",
        b"two",
        b"two",
        b"two",
    ]
