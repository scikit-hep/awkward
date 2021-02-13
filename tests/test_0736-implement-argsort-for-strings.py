# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


# def test_but_first_fix_sort():
#     assert ak.is_valid(ak.sort(ak.Array(["one", "two", "three"])))


def test_argsort():
    array = ak.Array(["one", "two", "three", "four", "five", "six", "seven", "eight"])
    assert ak.argsort(array, axis=0).tolist() == [7, 4, 3, 0, 6, 5, 2, 1]

    array = ak.Array(
        [["twotwo", "two", "three"], ["four", "five"], [], ["six", "seven", "eight"]]
    )
    assert ak.argsort(array, axis=1).tolist() == [[2, 1, 0], [1, 0], [], [2, 1, 0]]

    array = ak.Array(
        [[["twotwo", "two"], ["three"]], [["four", "five"]], [], [["six"], ["seven", "eight"]]]
    )
    assert ak.argsort(array, axis=2).tolist() == [[[1, 0], [0]], [[1, 0]], [], [[0], [1, 0]]]
