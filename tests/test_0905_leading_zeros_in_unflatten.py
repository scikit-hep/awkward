# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.Array([[1, 2, 3]])
    assert ak.operations.unflatten(array, [2, 1], axis=1).to_list() == [[[1, 2], [3]]]
    assert ak.operations.unflatten(array, [0, 2, 1], axis=1).to_list() == [
        [[], [1, 2], [3]]
    ]
    assert ak.operations.unflatten(array, [0, 0, 2, 1], axis=1).to_list() == [
        [[], [], [1, 2], [3]]
    ]
