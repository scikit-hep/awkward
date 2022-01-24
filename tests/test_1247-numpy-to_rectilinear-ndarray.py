# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = np.array([1, 2, 9, 0])
    nplike = ak.nplike.of(array)

    ak_array = ak.from_numpy(array)
    assert ak.to_list(nplike.to_rectilinear(array)) == ak.to_list(ak_array)
