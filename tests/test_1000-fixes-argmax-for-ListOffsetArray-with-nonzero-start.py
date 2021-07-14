# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_mine():
    array = ak.Array(
        ak.layout.ListOffsetArray64(
            ak.layout.Index64(np.array([1, 2, 4, 7], np.int64)),
            ak.layout.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
            ),
        )
    )
    assert (
        ak.argmax(array, axis=-1).tolist()
        == ak.argmax(ak.Array([[1.1], [2.2, 3.3], [4.4, 5.5, 6.6]]), axis=-1).tolist()
    )


def test_998():
    array = ak.Array(
        ak.layout.ListOffsetArray64(
            ak.layout.Index64(np.array([1, 2, 4, 7], np.int64)),
            ak.layout.NumpyArray(
                np.r_[
                    1.8125,
                    0.8125,
                    -0.9375,
                    1.1875,
                    -0.6875,
                    1.3125,
                    21.3125,
                ]
            ),
        )
    )

    assert ak.to_list(ak.argmax(array, axis=-1)) == [0, 1, 2]


def test_1000():
    array = ak.Array(
        ak.layout.ListOffsetArray64(
            ak.layout.Index64(np.array([1, 3, 5], np.int64)),
            ak.layout.ListOffsetArray64(
                ak.layout.Index64(np.array([0, 3, 5, 8, 10, 12], np.int64)),
                ak.layout.NumpyArray(
                    np.r_[
                        1.8125,
                        0.81252,
                        -0.937,
                        6.0,
                        -0.6875,
                        1.3125,
                        21.3125,
                        4.0,
                        9.8,
                        2.2,
                        33.0,
                        44.6,
                    ]
                ),
            ),
        )
    )

    assert ak.to_list(ak.argmax(array, axis=1)) == [[0, 1, 1], [1, 1]]
