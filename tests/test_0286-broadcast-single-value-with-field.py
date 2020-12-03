# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_broadcast_single_bool():
    base = ak.Array([[{"x": 0.1, "y": 0.2, "z": 0.3}, {"x": 0.4, "y": 0.5, "z": 0.6}]])
    base_new1 = ak.with_field(base, True, "always_true")
    assert ak.to_list(base_new1.always_true) == [[True, True]]
    base_new2 = ak.with_field(base_new1, base.x > 0.3, "sometimes_true")
    assert ak.to_list(base_new2.always_true) == [[True, True]]
