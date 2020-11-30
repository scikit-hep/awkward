# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = ak.Array([[1, 2, 3], [], [4, 5], [6, 7, 8, 9]])
    assert ak.flatten(array[:-1], axis=1).tolist() == [1, 2, 3, 4, 5]
    assert ak.flatten(array[:-2], axis=1).tolist() == [1, 2, 3]
    assert ak.flatten(array[:-1], axis=None).tolist() == [1, 2, 3, 4, 5]
    assert ak.flatten(array[:-2], axis=None).tolist() == [1, 2, 3]
