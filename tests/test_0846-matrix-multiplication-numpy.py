# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pytest.importorskip("numba")


def test_numpy_rhs():
    transform = ak.Array(
        [
            [0, 1, 0],
            [4, 0, 0],
            [0, 0, 2],
        ]
    )

    vector = ak.Array(np.r_[4, 5, 6].reshape(3, 1))

    result = np.matmul(transform, vector)

    assert result.tolist() == [[5], [16], [12]]


def test_numpy_lhs():
    transform = ak.Array(
        [
            [0, 4, 0],
            [1, 0, 0],
            [0, 0, 2],
        ]
    )

    vector = ak.Array(np.r_[4, 5, 6].reshape(1, 3))

    result = np.matmul(vector, transform)

    assert result.tolist() == [[5, 16, 12]]
