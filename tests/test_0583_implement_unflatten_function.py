# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert ak.operations.unflatten(array, 5).to_list() == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
    ]
    assert ak.operations.unflatten(array, [3, 0, 2, 1, 4]).to_list() == [
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
    ]
    assert ak.operations.unflatten(array, [3, None, 2, 1, 4]).to_list() == [
        [0, 1, 2],
        None,
        [3, 4],
        [5],
        [6, 7, 8, 9],
    ]

    original = ak.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]])
    counts = ak.operations.num(original)
    array = ak.operations.flatten(original)
    assert counts.to_list() == [3, 0, 2, 1, 4]
    assert array.to_list() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert ak.operations.unflatten(array, counts).to_list() == [
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
    ]
