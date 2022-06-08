# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    x = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 2, 2, 4, 6])),
        ak._v2.contents.NumpyArray(np.arange(8 * 8).reshape(8, -1)),
    )
    y = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 2, 2, 4, 6])),
        ak._v2.contents.NumpyArray(np.arange(8)),
    )
    u, v = ak._v2.operations.broadcast_arrays(x, y)
    assert u.ndim == v.ndim
