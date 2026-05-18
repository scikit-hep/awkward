from __future__ import annotations

import numpy as np

import awkward as ak


def test_ak_full_like_with_broadcasting():
    a = ak.Array(np.ones((2, 2)))
    b = ak.full_like(a, fill_value=np.array([2.0, 3.0]))

    assert ak.to_list(b) == [[2.0, 3.0], [2.0, 3.0]]
