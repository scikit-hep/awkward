# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_date_time():

    array = ak.Array(
        np.array(["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[s]")
    )
    assert array.tolist() == [1595846471, 1546300800, 1577836800]

    array1 = ak.Array(np.array(["2020-07-27T10:41:11.200000011"], "datetime64"))
    assert array1.tolist() == [1595846471200000011]
