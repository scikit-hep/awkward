# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2, v2_to_v1

def test_localindex():
    v1_array = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v1_array.localindex(0)) == ak.to_list(v2_to_v1(v2_array.localindex(0)))
    assert ak.to_list(v1_array.localindex(1)) == ak.to_list(v2_to_v1(v2_array.localindex(1)))
