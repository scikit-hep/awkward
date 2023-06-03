# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_reducer():
    # broadcast_tooffsets64 was changed, and stopped trimming the content
    # this test ensures that things still work!
    a = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(
                np.array([1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5], dtype=np.int64)
            ),
        )
    )
    # Perform broadcast, check result works!
    assert ak.all(a == [[1], [2]])
