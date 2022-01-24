# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    reference = np.r_[1, 2, 3, 4]
    test = ak.Array([1, 2, 9, 0])
    assert ak.to_list(np.isin(test, reference)) == [True, True, False, False]
