# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    left = ak.Array([1, 2, 3])
    right = ak.Array([[1, 2], [4, 5, 6], [None]])
    result = np.broadcast_arrays(left, right)
    assert isinstance(result, list)
    assert isinstance(result[0], ak.Array)
    assert isinstance(result[1], ak.Array)
    assert result[0].to_list() == [[1, 1], [2, 2, 2], [None]]
    assert result[1].to_list() == [[1, 2], [4, 5, 6], [None]]
