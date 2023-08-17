# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.Array([[1, 2, 3, 5, 7, 8, 10], [100000, 20000, 301]])
    quotient, remainder = np.divmod(array, 2)
    assert ak.almost_equal(quotient, [[0, 1, 1, 2, 3, 4, 5], [50000, 10000, 150]])
    assert ak.almost_equal(remainder, [[1, 0, 1, 1, 1, 0, 0], [0, 0, 1]])
