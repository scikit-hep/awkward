# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    x = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2, 2, 4, 6])),
        ak.contents.NumpyArray(np.arange(8 * 8).reshape(8, -1)),
    )
    y = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2, 2, 4, 6])),
        ak.contents.NumpyArray(np.arange(8)),
    )
    u, v = ak.operations.broadcast_arrays(x, y)
    assert u.ndim == v.ndim
