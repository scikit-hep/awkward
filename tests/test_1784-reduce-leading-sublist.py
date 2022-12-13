# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


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

    reduced = ak.sum(small, axis=-2)
    assert reduced.to_list() == [[], [4, 6]]
    assert np.asarray(reduced.layout.starts).tolist() == [0, 2]
    assert np.asarray(reduced.layout.stops).tolist() == [0, 4]
