# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_2d():
    array = ak.from_iter(
        [
            [3.3, 2.2, 5.5, 1.1, 4.4],
            [4.4, 2.2, 1.1, 3.3, 5.5],
            [2.2, 1.1, 4.4, 3.3, 5.5],
        ],
        highlevel=False,
    )
    assert ak.to_list(array.argmin(axis=0)) == [2, 2, 1, 0, 0]
    assert ak.to_list(array.argmin(axis=1)) == [3, 2, 1]


def test_3d():
    array = ak.from_iter(
        [
            [
                [3.3, 2.2, 5.5, 1.1, 4.4],
                [4.4, 2.2, 1.1, 3.3, 5.5],
                [2.2, 1.1, 4.4, 3.3, 5.5],
            ],
            [
                [-3.3, 2.2, -5.5, 1.1, 4.4],
                [4.4, -2.2, 1.1, -3.3, 5.5],
                [2.2, 1.1, 4.4, 3.3, -5.5],
            ],
        ],
        highlevel=False,
    )
    assert ak.to_list(array.argmin(axis=0)) == [
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
    assert ak.to_list(array.argmin(axis=1)) == [[2, 2, 1, 0, 0], [0, 1, 0, 1, 2]]
    assert ak.to_list(array.argmin(axis=2)) == [[3, 2, 1], [2, 3, 4]]
    assert ak.to_list(array.argmin(axis=-1)) == [[3, 2, 1], [2, 3, 4]]
    assert ak.to_list(array.argmin(axis=-2)) == [[2, 2, 1, 0, 0], [0, 1, 0, 1, 2]]
    assert ak.to_list(array.argmin(axis=-3)) == [
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]


def test_jagged():
    array = ak.from_iter(
        [[2.2, 1.1, 3.3], [], [4.4, 5.5], [5.5], [-4.4, -5.5, -6.6]], highlevel=False
    )
    assert ak.to_list(array.argmin(axis=1)) == [1, None, 0, 0, 2]

    index2 = ak.layout.Index64(np.array([4, 3, 2, 1, 0], dtype=np.int64))
    array2 = ak.layout.IndexedArray64(index2, array)
    assert ak.to_list(array2.argmin(axis=1)) == [2, 0, 0, None, 1]

    index3 = ak.layout.Index64(np.array([4, 3, -1, 4, 0], dtype=np.int64))
    array2 = ak.layout.IndexedArray64(index3, array)
    assert ak.to_list(array2.argmin(axis=1)) == [2, 0, None, 2, 1]
    assert ak.to_list(array2.argmin(axis=-1)) == [2, 0, None, 2, 1]


def test_missing():
    array = ak.from_iter(
        [[[2.2, 1.1, 3.3]], [[]], [None, None, None], [[-4.4, -5.5, -6.6]]],
        highlevel=False,
    )
    assert ak.to_list(array.argmin(axis=2)) == [[1], [None], [None, None, None], [2]]


def test_highlevel():
    array = ak.Array(
        [[3.3, 1.1, 5.5, 1.1, 4.4], [4.4, 2.2, 1.1, 6.6], [2.2, 3.3, -1.1]]
    )
    assert ak.argmin(array) == 11
    assert ak.argmax(array) == 8
    assert ak.to_list(ak.argmin(array, axis=0)) == [2, 0, 2, 0, 0]
    assert ak.to_list(ak.argmax(array, axis=0)) == [1, 2, 0, 1, 0]
    assert ak.to_list(ak.argmin(array, axis=1)) == [1, 2, 2]
    assert ak.to_list(ak.argmax(array, axis=1)) == [2, 3, 1]
