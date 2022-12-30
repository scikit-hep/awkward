# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_numpy_array():
    x = np.arange(12)
    y = ak.operations.is_none(x)

    assert y.to_list() == [False] * 12


def test_awkward_from_numpy_array():
    x = np.arange(12)
    y = ak.operations.from_numpy(x)
    z = ak.operations.is_none(y)

    assert z.to_list() == [False] * 12
