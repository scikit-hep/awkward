# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

from awkward.types.numpytype import dtype_to_primitive


def test_time_dtypes_with_equal_hashes():
    assert dtype_to_primitive(np.dtype("m8[3000ps]")) == "timedelta64[3000ps]"
    assert dtype_to_primitive(np.dtype("m8[3ns]")) == "timedelta64[3ns]"
    assert dtype_to_primitive(np.dtype("M8[s]")) == "datetime64[s]"
    assert dtype_to_primitive(np.dtype("M8[0s]")) == "datetime64[0s]"
