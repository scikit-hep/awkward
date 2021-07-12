# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = ak.Array([[0, 1, 2, 3], [3, 3, 3, 2, 1]])
    is_valid = array != 3

    assert array.mask[is_valid].tolist() == [[0, 1, 2, None], [None, None, None, 2, 1]]

    assert ak.sort(array.mask[is_valid]).tolist() == [
        [0, 1, 2, None],
        [1, 2, None, None, None],
    ]
    assert ak.argsort(array.mask[is_valid]).tolist() == [
        [0, 1, 2, 3],
        [4, 3, 0, 1, 2],
    ]
