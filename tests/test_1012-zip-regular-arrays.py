# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    parameters = ak.from_numpy(np.arange(3 * 3).reshape(-1, 3), regulararray=True)
    array = ak.zip((parameters, parameters))
    assert array.layout.purelist_isregular

    assert ak.to_list(array) == [
        [(0, 0), (1, 1), (2, 2)],
        [(3, 3), (4, 4), (5, 5)],
        [(6, 6), (7, 7), (8, 8)],
    ]
