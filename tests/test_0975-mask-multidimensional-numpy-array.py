# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = ak.Array(
        ak.layout.RegularArray(
            ak.layout.NumpyArray(np.r_[1, 2, 3, 4, 5, 6, 7, 8, 9]), 3
        )
    )
    mask = ak.Array(
        ak.layout.NumpyArray(
            np.array([[True, True, True], [True, True, False], [True, False, True]])
        )
    )

    assert ak.mask(array, mask).to_list() == [[1, 2, 3], [4, 5, None], [7, None, 9]]
