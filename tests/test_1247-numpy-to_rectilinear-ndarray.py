# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    array = np.array([1, 2, 9, 0])
    nplike = ak._nplikes.nplike_of(array)
    ak_array = ak.operations.from_numpy(array)
    assert nplike.to_rectilinear(array).tolist() == ak_array.to_list()
