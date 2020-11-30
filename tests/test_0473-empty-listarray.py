# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward1 as ak  # noqa: F401


def test_empty_listarray():
    a = ak.Array(
        ak.layout.ListArray64(
            ak.layout.Index64(np.array([], dtype=np.int64)),
            ak.layout.Index64(np.array([], dtype=np.int64)),
            ak.layout.NumpyArray(np.array([])),
        )
    )
    assert ak.to_list(a * 3) == []

    starts = ak.layout.Index64(np.array([], dtype=np.int64))
    stops = ak.layout.Index64(np.array([3, 3, 5], dtype=np.int64))
    content = ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    array = ak.Array(ak.layout.ListArray64(starts, stops, content))
    array + array
