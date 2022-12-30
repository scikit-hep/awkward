# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test_sort():
    data = ak.Array([[7, 5, 7], [], [2], [8, 2]])
    assert to_list(ak.operations.sort(data)) == [
        [5, 7, 7],
        [],
        [2],
        [2, 8],
    ]


def test_argsort():
    data = ak.Array([[7, 5, 7], [], [2], [8, 2]])
    index = ak.operations.argsort(data)
    assert to_list(data[index]) == [[5, 7, 7], [], [2], [2, 8]]
