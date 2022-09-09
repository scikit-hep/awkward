# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test_sort():
    data = ak._v2.Array([[7, 5, 7], [], [2], [8, 2]])
    assert to_list(ak._v2.operations.sort(data)) == [
        [5, 7, 7],
        [],
        [2],
        [2, 8],
    ]


def test_argsort():
    data = ak._v2.Array([[7, 5, 7], [], [2], [8, 2]])
    index = ak._v2.operations.argsort(data)
    assert to_list(data[index]) == [[5, 7, 7], [], [2], [2, 8]]
