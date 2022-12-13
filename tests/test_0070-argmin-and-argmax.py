# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test_1d():
    array = ak.operations.from_iter(
        [3.3, 2.2, 5.5, 1.1, 4.4],
        highlevel=False,
    )
    assert to_list(ak.argmin(array, axis=0, highlevel=False)) == 3
    assert to_list(ak.argmax(array, axis=0, highlevel=False)) == 2
    assert to_list(ak.count(array, axis=0, highlevel=False)) == 5
    assert to_list(ak.count_nonzero(array, axis=0, highlevel=False)) == 5
    assert to_list(ak.sum(array, axis=0, highlevel=False)) == 16.5
    assert to_list(ak.prod(array, axis=0, highlevel=False)) == 193.26120000000003
    assert to_list(ak.any(array, axis=0, highlevel=False)) is True
    assert to_list(ak.all(array, axis=0, highlevel=False)) is True
    assert to_list(ak.min(array, axis=0, highlevel=False)) == 1.1
    assert to_list(ak.max(array, axis=0, highlevel=False)) == 5.5


def test_2d():
    array = ak.operations.from_iter(
        [
            [3.3, 2.2, 5.5, 1.1, 4.4],
            [4.4, 2.2, 1.1, 3.3, 5.5],
            [2.2, 1.1, 4.4, 3.3, 5.5],
        ],
        highlevel=False,
    )
    assert to_list(ak.argmin(array, axis=0, highlevel=False)) == [2, 2, 1, 0, 0]
    assert (
        ak.argmin(array.to_typetracer(), axis=0, highlevel=False).form
        == ak.argmin(array, axis=0, highlevel=False).form
    )
    assert to_list(ak.argmin(array, axis=1, highlevel=False)) == [3, 2, 1]
    assert (
        ak.argmin(array.to_typetracer(), axis=1, highlevel=False).form
        == ak.argmin(array, axis=1, highlevel=False).form
    )


def test_3d():
    array = ak.operations.from_iter(
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

    assert to_list(ak.argmin(array, axis=0, highlevel=False)) == [
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
    assert to_list(ak.argmin(array, axis=1, highlevel=False)) == [
        [2, 2, 1, 0, 0],
        [0, 1, 0, 1, 2],
    ]
    assert (
        ak.argmin(array.to_typetracer(), axis=1, highlevel=False).form
        == ak.argmin(array, axis=1, highlevel=False).form
    )
    assert to_list(ak.argmin(array, axis=2, highlevel=False)) == [[3, 2, 1], [2, 3, 4]]
    assert (
        ak.argmin(array.to_typetracer(), axis=2, highlevel=False).form
        == ak.argmin(array, axis=2, highlevel=False).form
    )
    assert to_list(ak.argmin(array, axis=-1, highlevel=False)) == [[3, 2, 1], [2, 3, 4]]
    assert (
        ak.argmin(array.to_typetracer(), axis=-1, highlevel=False).form
        == ak.argmin(array, axis=-1, highlevel=False).form
    )
    assert to_list(ak.argmin(array, axis=-2, highlevel=False)) == [
        [2, 2, 1, 0, 0],
        [0, 1, 0, 1, 2],
    ]
    assert (
        ak.argmin(array.to_typetracer(), axis=-2, highlevel=False).form
        == ak.argmin(array, axis=-2, highlevel=False).form
    )
    assert to_list(ak.argmin(array, axis=-3, highlevel=False)) == [
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
    assert (
        ak.argmin(array.to_typetracer(), axis=-3, highlevel=False).form
        == ak.argmin(array, axis=-3, highlevel=False).form
    )


def test_jagged():
    v2_array = ak.operations.from_iter(
        [[2.2, 1.1, 3.3], [], [4.4, 5.5], [5.5], [-4.4, -5.5, -6.6]], highlevel=False
    )
    assert to_list(ak.argmin(v2_array, axis=1, highlevel=False)) == [1, None, 0, 0, 2]
    assert (
        ak.argmin(v2_array.to_typetracer(), axis=1, highlevel=False).form
        == ak.argmin(v2_array, axis=1, highlevel=False).form
    )

    index2 = ak.index.Index64(np.array([4, 3, 2, 1, 0], dtype=np.int64))
    v2_array2 = ak.contents.IndexedArray(index2, v2_array)
    assert to_list(ak.argmin(v2_array2, axis=1, highlevel=False)) == [2, 0, 0, None, 1]
    assert (
        ak.argmin(v2_array2.to_typetracer(), axis=1, highlevel=False).form
        == ak.argmin(v2_array2, axis=1, highlevel=False).form
    )

    index3 = ak.index.Index64(np.array([4, 3, -1, 4, 0], dtype=np.int64))
    v2_array2 = ak.contents.IndexedOptionArray(index3, v2_array)
    assert to_list(ak.argmin(v2_array2, axis=1, highlevel=False)) == [2, 0, None, 2, 1]
    assert (
        ak.argmin(v2_array2.to_typetracer(), axis=1, highlevel=False).form
        == ak.argmin(v2_array2, axis=1, highlevel=False).form
    )
    assert to_list(ak.argmin(v2_array2, axis=-1, highlevel=False)) == [2, 0, None, 2, 1]
    assert (
        ak.argmin(v2_array2.to_typetracer(), axis=-1, highlevel=False).form
        == ak.argmin(v2_array2, axis=-1, highlevel=False).form
    )


def test_missing():
    array = ak.operations.from_iter(
        [[[2.2, 1.1, 3.3]], [[]], [None, None, None], [[-4.4, -5.5, -6.6]]],
        highlevel=False,
    )
    assert to_list(ak.argmin(array, axis=2, highlevel=False)) == [
        [1],
        [None],
        [None, None, None],
        [2],
    ]
    assert (
        ak.argmin(array.to_typetracer(), axis=2, highlevel=False).form
        == ak.argmin(array, axis=2, highlevel=False).form
    )
