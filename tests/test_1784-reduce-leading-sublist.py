# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401


def test():
    small = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 0, 2])),
            ak.contents.ListOffsetArray(
                ak.index.Index64(np.array([0, 2, 4])),
                ak.contents.NumpyArray(np.array([1, 2, 3, 4], dtype=np.float32)),
            ),
        )
    )

    assert ak.sum(small, axis=-2).tolist() == [[], [4, 6]]
