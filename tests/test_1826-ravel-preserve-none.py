# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak
import awkward._connect.cling
import awkward._lookup


def test():
    array = ak.Array([[1, 2, 3, None], [4, 5, 6, 7, 8], [], [9], None, [10]])

    assert ak.ravel(array).to_list() == [1, 2, 3, None, 4, 5, 6, 7, 8, 9, 10]
    assert ak.flatten(array, axis=None).to_list() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
