# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test():
    scalar = ak.Array(np.arange(5, dtype=np.int64), backend="typetracer")[0]
    result = scalar.view(np.float64)
    assert result.dtype == np.dtype(np.float64)
    assert result.ndim == 0
    with pytest.raises(RuntimeError):
        int(result)
