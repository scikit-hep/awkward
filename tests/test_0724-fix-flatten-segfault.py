# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    a = ak.layout.NumpyArray(np.empty(0))
    idx = ak.layout.Index64([])
    a = ak.layout.IndexedOptionArray64(idx, a)
    idx = ak.layout.Index64([0])
    a = ak.layout.ListOffsetArray64(idx, a)
    idx = ak.layout.Index64([175990832])
    a = ak.layout.ListOffsetArray64(idx, a)
    assert ak.flatten(a, axis=2).tolist() == []
    assert str(ak.flatten(a, axis=2).type) == "0 * var * ?float64"
