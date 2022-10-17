# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401


def test():
    small = ak._v2.Array(
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([0, 0, 2])),
            ak._v2.contents.ListOffsetArray(
                ak._v2.index.Index64(np.array([0, 2, 4])),
                ak._v2.contents.NumpyArray(np.array([1, 2, 3, 4], dtype=np.float32)),
            ),
        )
    )

    reduced = ak._v2.sum(small, axis=-2)
    assert reduced.tolist() == [[], [4, 6]]
    assert np.asarray(reduced.layout.starts).tolist() == [0, 2]
    assert np.asarray(reduced.layout.stops).tolist() == [0, 4]
