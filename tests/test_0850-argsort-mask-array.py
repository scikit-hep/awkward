# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.Array([[0, 1, 2, 3], [3, 3, 3, 2, 1]])
    is_valid = array != 3

    assert ak.operations.mask(array, is_valid).to_list() == [
        [0, 1, 2, None],
        [None, None, None, 2, 1],
    ]

    assert ak.operations.sort(ak.operations.mask(array, is_valid)).to_list() == [
        [0, 1, 2, None],
        [1, 2, None, None, None],
    ]
    assert ak.operations.argsort(ak.operations.mask(array, is_valid)).to_list() == [
        [0, 1, 2, 3],
        [4, 3, 0, 1, 2],
    ]
