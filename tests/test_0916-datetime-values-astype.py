# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_values_astype_datetime():
    array1 = ak.values_astype(ak.Array([1567416600000]), "datetime64[ms]")
    assert str(array1.type) == "1 * datetime64"
    assert array1.to_list() == [np.datetime64("2019-09-02T09:30:00")]

    array2 = ak.values_astype(ak.Array([1567416600000]), np.dtype("M8[ms]"))
    assert str(array2.type) == "1 * datetime64"
    assert array2.to_list() == [np.datetime64("2019-09-02T09:30:00")]

    array3 = ak.values_astype(
        ak.Array([1567416600000000, None]), np.datetime64  # default unit is 'us'
    )
    assert array3.to_list() == [np.datetime64("2019-09-02T09:30:00"), None]
