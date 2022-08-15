# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_numpy_array():
    x = np.arange(12)
    y = ak._v2.operations.is_none(x)

    assert y.tolist() == [False] * 12


def test_awkward_from_numpy_array():
    x = np.arange(12)
    y = ak._v2.operations.from_numpy(x)
    z = ak._v2.operations.is_none(y)

    assert z.tolist() == [False] * 12
