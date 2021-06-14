# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_date_time_values_astype():
    array = ak.values_astype(ak.Array([1567416600000]), np.datetime64)
    assert str(array.type) == "1 * datetime64"
