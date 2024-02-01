# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test():
    data = ak.Array(
        [[[0, 1.1, 2.2], []], [], [[3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]]
    )
    assert ak.var(data, axis=0).tolist() == [
        pytest.approx([2.7225, 2.7225, 0]),
        pytest.approx([0]),
        pytest.approx([0, 0, 0, 0]),
    ]
    assert ak.var(data, axis=1).tolist() == [
        pytest.approx([0, 0, 0]),
        pytest.approx([]),
        pytest.approx([1.88222222, 2.7225, 0, 0]),
    ]
    assert ak.var(data, axis=2).tolist() == [
        pytest.approx([0.80666667, np.nan], nan_ok=True),
        pytest.approx([]),
        pytest.approx([0.3025, 0, 1.5125]),
    ]
