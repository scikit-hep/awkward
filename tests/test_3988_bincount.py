from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_awkward_numpy_bincount_basic():
    data = ak.Array([0, 1, 1, 2, 2, 2])
    result = np.bincount(data)

    assert isinstance(result, ak.Array)
    assert np.array_equal(ak.to_numpy(result), [1, 2, 3])


def test_awkward_numpy_bincount_with_minlength():
    data = ak.Array([0, 1, 2])
    result = np.bincount(data, minlength=5)

    assert len(result) == 5
    assert np.array_equal(ak.to_numpy(result), [1, 1, 1, 0, 0])


def test_awkward_numpy_bincount_with_weights():
    data = ak.Array([0, 1, 2])
    weights = ak.Array([0.5, 0.5, 1.0])

    result = np.bincount(data, weights=weights)

    assert result.layout.dtype == np.float64
    assert np.array_equal(ak.to_numpy(result), [0.5, 0.5, 1.0])


def test_awkward_masked_array_bincount():
    data = ak.Array([0, 1, None, 2])

    result = np.bincount(data)

    assert isinstance(result, ak.Array)

    expected = []
    assert np.array_equal(ak.to_numpy(result), expected)


def test_awkward_jagged_bincount_fails():
    jagged = ak.Array([[0, 1], [1, 2, 2]])

    with pytest.raises(ValueError):
        np.bincount(jagged)

    result = np.bincount(ak.flatten(jagged))
    assert np.array_equal(ak.to_numpy(result), [1, 2, 2])
