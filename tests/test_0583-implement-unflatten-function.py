# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = ak.Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert ak.unflatten(array, 5).tolist() == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert ak.unflatten(array, [3, 0, 2, 1, 4]).tolist() == [
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
    ]
    assert ak.unflatten(array, [3, None, 2, 1, 4]).tolist() == [
        [0, 1, 2],
        None,
        [3, 4],
        [5],
        [6, 7, 8, 9],
    ]

    original = ak.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]])
    counts = ak.num(original)
    array = ak.flatten(original)
    assert counts.tolist() == [3, 0, 2, 1, 4]
    assert array.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert ak.unflatten(array, counts).tolist() == [
        [0, 1, 2],
        [],
        [3, 4],
        [5],
        [6, 7, 8, 9],
    ]
