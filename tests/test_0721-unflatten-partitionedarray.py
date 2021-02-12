# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    content = ak.repartition(range(10), 3)
    assert ak.unflatten(content, [3, 0, 2, 1, 3, 1]).tolist() == [
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8],
        [9],
    ]

    original = ak.repartition(ak.Array([[1, 2, 3, 4], [5, 6, 7], [8, 9]]), 2)

    assert ak.unflatten(original, [2, 2, 1, 2, 1, 1], axis=1).tolist() == [
        [[1, 2], [3, 4]],
        [[5], [6, 7]],
        [[8], [9]],
    ]

    original = ak.repartition(ak.Array([[1, 2, 3, 4], [], [5, 6, 7], [8, 9]]), 2)
    assert ak.unflatten(original, [2, 2, 1, 2, 1, 1], axis=1).tolist() == [
        [[1, 2], [3, 4]],
        [],
        [[5], [6, 7]],
        [[8], [9]],
    ]
