# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_regular():
    i22 = np.array([[1, 0], [0, 1]])
    ii22 = np.array([[2, 0], [0, 2]])
    iii22 = np.array([[3, 0], [0, 3]])
    a22 = np.array([[1, 2], [3, 4]])
    b22 = np.array([[5, 6], [7, 8]])

    a23 = np.array([[1, 2, 3], [4, 5, 6]])
    b23 = np.array([[7, 8, 9], [10, 11, 12]])

    a32 = np.array([[1, 2], [3, 4], [5, 6]])
    b32 = np.array([[7, 8], [9, 10], [11, 12]])

    i33 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ii33 = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    iii33 = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    a33 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b33 = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])

    assert (
        np.matmul(a22, b22).tolist() == np.matmul(ak.Array(a22), ak.Array(b22)).tolist()
    )
    assert (
        np.matmul(a33, b33).tolist() == np.matmul(ak.Array(a33), ak.Array(b33)).tolist()
    )
    assert (
        np.matmul(a22, b23).tolist() == np.matmul(ak.Array(a22), ak.Array(b23)).tolist()
    )
    assert (
        np.matmul(a23, b33).tolist() == np.matmul(ak.Array(a23), ak.Array(b33)).tolist()
    )
    assert (
        np.matmul(a33, b32).tolist() == np.matmul(ak.Array(a33), ak.Array(b32)).tolist()
    )

    assert (
        np.matmul(
            np.array([a22, i22, i22, a22]), np.array([b22, i22, b22, i22])
        ).tolist()
        == np.matmul(
            ak.Array(np.array([a22, i22, i22, a22])),
            ak.Array(np.array([b22, i22, b22, i22])),
        ).tolist()
    )


numba = pytest.importorskip("numba")


def test_irregular():
    i22 = np.array([[1, 0], [0, 1]])
    a22 = np.array([[1, 2], [3, 4]])
    b22 = np.array([[5, 6], [7, 8]])

    assert (
        np.matmul(
            np.array([a22, i22, i22, a22]), np.array([b22, i22, b22, i22])
        ).tolist()
        == np.matmul(
            ak.Array([a22, i22, i22, a22]), ak.Array([b22, i22, b22, i22])
        ).tolist()
    )

    lefts = ak.Array(
        [[[1, 2], [3, 4], [5, 6]], [[1, 2, 3, 4], [5, 6, 7, 8]], [[1], [2], [3], [4]],]
    )
    rights = ak.Array(
        [
            [[7, 8, 9], [10, 11, 12]],
            [[8, 10], [11, 12], [13, 14], [15, 16]],
            [[5, 6, 7]],
        ]
    )

    assert np.matmul(lefts, rights).tolist() == [
        [[27, 30, 33], [61, 68, 75], [95, 106, 117]],
        [[129, 140], [317, 348]],
        [[5, 6, 7], [10, 12, 14], [15, 18, 21], [20, 24, 28]],
    ]
