# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_values_astype_datetime():
    array = ak.values_astype(
        ak.Array([1567416600000]), "datetime64[ms]"
    )  # np.datetime64)
    assert str(array.type) == "1 * datetime64"
    assert array.to_list() == [np.datetime64("2019-09-02T09:30:00")]

    # arr = ak.Array(['2019-09-02T09:30:00',  None])
    arr = ak.Array([1567416600000000, None])  # default unit is 'us'
    dt_arr = ak.values_astype(arr, np.datetime64)
    assert dt_arr.to_list() == [np.datetime64("2019-09-02T09:30:00"), None]
