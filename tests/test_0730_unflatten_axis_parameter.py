# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak


def test():
    original = ak.Array([[1, 2, 3, 4], [], [5, 6, 7], [8, 9]])

    assert ak.operations.unflatten(original, [2, 2, 1, 2, 1, 1], axis=1).to_list() == [
        [[1, 2], [3, 4]],
        [],
        [[5], [6, 7]],
        [[8], [9]],
    ]

    assert ak.operations.unflatten(original, [1, 3, 1, 2, 1, 1], axis=1).to_list() == [
        [[1], [2, 3, 4]],
        [],
        [[5], [6, 7]],
        [[8], [9]],
    ]

    with pytest.raises(ValueError):
        ak.operations.unflatten(original, [2, 1, 2, 2, 1, 1], axis=1)

    assert ak.operations.unflatten(
        original, [2, 0, 2, 1, 2, 1, 1], axis=1
    ).to_list() == [
        [[1, 2], [], [3, 4]],
        [],
        [[5], [6, 7]],
        [[8], [9]],
    ]


def test_issue742():
    assert ak.operations.unflatten(ak.Array(["a", "b", "c"]), [1, 2, 0]).to_list() == [
        ["a"],
        ["b", "c"],
        [],
    ]
