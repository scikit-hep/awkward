# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    assert (
        ak.Array([[1, 2], [3, 4]])[[True, False], 0].ndim
        == np.array([[1, 2], [3, 4]])[[True, False], 0].ndim
    )
    assert (
        ak.Array([[1, 2], [3, 4]])[[False, False], 0].ndim
        == np.array([[1, 2], [3, 4]])[[False, False], 0].ndim
    )

    assert (
        ak.Array([[1, 2], [3, 4]])[[True, False], 0].ndim
        == np.array([[1, 2], [3, 4]])[[True, False]][:, 0].ndim
    )
    assert (
        ak.Array([[1, 2], [3, 4]])[[False, False], 0].ndim
        == np.array([[1, 2], [3, 4]])[[False, False]][:, 0].ndim
    )
