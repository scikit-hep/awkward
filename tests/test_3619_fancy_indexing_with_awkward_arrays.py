# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
#
from __future__ import annotations

import pytest

import awkward as ak


def test():
    # taking the array directly from the issue: https://github.com/scikit-hep/awkward/issues/3619
    a = ak.Array([[1, 5], [3], [4]])

    # this is a valid case, where the index is not out of bounds
    c_valid = ak.Array([[1, 1], [0], [0]])
    assert a[c_valid].tolist() == [[5, 5], [3], [4]]

    # this is an invalid case, where the index for the second element `[1]` is out of bounds
    c_oob = ak.Array([[1, 1], [1], [0]])
    with pytest.raises(IndexError):
        a[c_oob]
