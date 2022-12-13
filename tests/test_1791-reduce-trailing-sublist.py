# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(
                np.array(
                    [0, 1, 1],
                    dtype=np.int64,
                )
            ),
            ak.contents.ListOffsetArray(
                ak.index.Index64(
                    np.array(
                        [0, 1],
                        dtype=np.int64,
                    )
                ),
                ak.contents.NumpyArray(np.arange(1)),
            ),
        )
    )

    reduced = ak.sum(array, axis=1)
    # We currently get a ListArray here. Ensure that the start/stops are correct
    assert np.asarray(reduced.layout.starts).tolist() == [0, 1]
    assert np.asarray(reduced.layout.stops).tolist() == [1, 1]


def test_review_examples():
    array = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 4, 4, 6]),
            ak.contents.ListOffsetArray(
                ak.index.Index64([0, 3, 3, 5, 6, 8, 9]),
                ak.contents.NumpyArray([2, 3, 5, 7, 11, 13, 17, 19, 23]),
            ),
        )
    )
    assert ak.Array(ak.prod(array, axis=-2), check_valid=True).to_list() == [
        [182, 33, 5],
        [],
        [391, 19],
    ]

    array = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 4, 4, 6]) + 2),
            ak.contents.ListOffsetArray(
                ak.index.Index64(np.array([123, 321, 0, 3, 3, 5, 6, 8, 9]) + 1),
                ak.contents.NumpyArray([9999, 2, 3, 5, 7, 11, 13, 17, 19, 23]),
            ),
        )
    )
    assert ak.Array(ak.prod(array, axis=-2), check_valid=True).to_list() == [
        [182, 33, 5],
        [],
        [391, 19],
    ]

    array = ak.Array(
        ak.contents.ListArray(
            ak.index.Index64([0, 4, 4]),
            ak.index.Index64([4, 4, 6]),
            ak.contents.ListArray(
                ak.index.Index64([0, 3, 3, 5, 6, 8]),
                ak.index.Index64([3, 3, 5, 6, 8, 9]),
                ak.contents.NumpyArray([2, 3, 5, 7, 11, 13, 17, 19, 23]),
            ),
        )
    )
    assert ak.Array(ak.prod(array, axis=-2), check_valid=True).to_list() == [
        [182, 33, 5],
        [],
        [391, 19],
    ]

    array = ak.Array(
        ak.contents.ListArray(
            ak.index.Index64([0, 100, 4]),
            ak.index.Index64([4, 100, 6]),
            ak.contents.ListArray(
                ak.index.Index64([0, 1000, 3, 5, 6, 8]),
                ak.index.Index64([3, 1000, 5, 6, 8, 9]),
                ak.contents.NumpyArray([2, 3, 5, 7, 11, 13, 17, 19, 23]),
            ),
        )
    )
    assert ak.Array(ak.prod(array, axis=-2), check_valid=True).to_list() == [
        [182, 33, 5],
        [],
        [391, 19],
    ]

    array = ak.Array(
        ak.contents.ListArray(
            ak.index.Index64([0, 100, 6]),
            ak.index.Index64([4, 100, 8]),
            ak.contents.ListArray(
                ak.index.Index64([0, 1000, 3, 5, 60, 80, 6, 8]),
                ak.index.Index64([3, 1000, 5, 6, 80, 90, 8, 9]),
                ak.contents.NumpyArray([2, 3, 5, 7, 11, 13, 17, 19, 23]),
            ),
        )
    )
    assert ak.Array(ak.prod(array, axis=-2), check_valid=True).to_list() == [
        [182, 33, 5],
        [],
        [391, 19],
    ]

    array = ak.Array(
        ak.contents.ListArray(
            ak.index.Index64([0, 100, 6]),
            ak.index.Index64([4, 100, 8]),
            ak.contents.ListArray(
                ak.index.Index64([9, 1000, 3, 5, 60, 80, 6, 8]),
                ak.index.Index64([12, 1000, 5, 6, 80, 90, 8, 9]),
                ak.contents.NumpyArray([20, 30, 50, 7, 11, 13, 17, 19, 23, 2, 3, 5]),
            ),
        )
    )
    assert ak.Array(ak.prod(array, axis=-2), check_valid=True).to_list() == [
        [182, 33, 5],
        [],
        [391, 19],
    ]
