# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_records_almost_equal():
    first = ak.contents.RecordArray(
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.dtype("<M8[s]"))),
        ],
        ["x", "y"],
    )

    second = ak.contents.RecordArray(
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.dtype("<M8[s]"))),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
        ],
        ["y", "x"],
    )

    assert ak.almost_equal(first, second)


def test_unions_almost_equal():
    # Check unions agree!
    first = ak.contents.UnionArray(
        ak.index.Index8([0, 0, 2, 1, 1]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("<M8[s]"))),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    second = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 0, 2, 2]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("<M8[s]"))),
        ],
    )
    assert ak.almost_equal(first, second)
