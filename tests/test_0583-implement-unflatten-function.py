# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401


def test():
    array = ak.Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert ak.operations.unflatten(array, 5).tolist() == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
    ]
    assert ak.operations.unflatten(array, [3, 0, 2, 1, 4]).tolist() == [
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
    ]
    assert ak.operations.unflatten(array, [3, None, 2, 1, 4]).tolist() == [
        [0, 1, 2],
        None,
        [3, 4],
        [5],
        [6, 7, 8, 9],
    ]

    original = ak.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]])
    counts = ak.operations.num(original)
    array = ak.operations.flatten(original)
    assert counts.tolist() == [3, 0, 2, 1, 4]
    assert array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert ak.operations.unflatten(array, counts).tolist() == [
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
    ]
