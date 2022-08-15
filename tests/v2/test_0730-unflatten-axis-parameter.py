# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    original = ak._v2.Array([[1, 2, 3, 4], [], [5, 6, 7], [8, 9]])

    assert ak._v2.operations.unflatten(
        original, [2, 2, 1, 2, 1, 1], axis=1
    ).tolist() == [
        [[1, 2], [3, 4]],
        [],
        [[5], [6, 7]],
        [[8], [9]],
    ]

    assert ak._v2.operations.unflatten(
        original, [1, 3, 1, 2, 1, 1], axis=1
    ).tolist() == [
        [[1], [2, 3, 4]],
        [],
        [[5], [6, 7]],
        [[8], [9]],
    ]

    with pytest.raises(ValueError):
        ak._v2.operations.unflatten(original, [2, 1, 2, 2, 1, 1], axis=1)

    assert ak._v2.operations.unflatten(
        original, [2, 0, 2, 1, 2, 1, 1], axis=1
    ).tolist() == [
        [[1, 2], [], [3, 4]],
        [],
        [[5], [6, 7]],
        [[8], [9]],
    ]


def test_issue742():
    assert ak._v2.operations.unflatten(
        ak._v2.Array(["a", "b", "c"]), [1, 2, 0]
    ).tolist() == [
        ["a"],
        ["b", "c"],
        [],
    ]
