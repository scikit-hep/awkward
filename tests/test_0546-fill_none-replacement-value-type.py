# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = ak.values_astype(ak.Array([1.1, 2.2, None, 3.3]), np.float32)
    assert str(ak.fill_none(array, np.float32(0)).type) == "4 * float32"
