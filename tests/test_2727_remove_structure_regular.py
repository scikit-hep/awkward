# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test():
    array = ak.Array([[[1, 2], [1]], [[1], [1], [1]]])
    assert ak.almost_equal(
        ak.mean(array, axis=None, keepdims=True),
        ak.contents.RegularArray(
            ak.contents.RegularArray(
                ak.contents.NumpyArray(
                    np.mean([1, 2, 1, 1, 1, 1], dtype=np.float64, keepdims=True)
                ),
                size=1,
            ),
            size=1,
        ),
        check_regular=True,
    )


def test_original_problem():
    array = ak.Array([[[1, 2], [1]], [[1], [1], [1]]])
    assert ak.mean(array) == pytest.approx(1.1666666666666667)
    assert ak.var(array) == pytest.approx(0.13888888888888887)
