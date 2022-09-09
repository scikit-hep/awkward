# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    x = ak._v2.Array([[0, 1], [], [3], [5], [6, 8, 9]])
    y = ak._v2.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]])

    with pytest.raises(ValueError):
        ak._v2.broadcast_arrays(x, y, depth_limit=None)

    u, v = ak._v2.broadcast_arrays(x, y, depth_limit=1)
    assert ak._v2.to_list(u) == [[0, 1], [], [3], [5], [6, 8, 9]]
    assert ak._v2.to_list(v) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
