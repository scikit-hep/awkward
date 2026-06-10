# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""
Tests for miscellaneous integration bug fixes:
- hist.py: broadcast_and_flatten returns NotImplemented (not NotImplementedError) on TypeError
"""

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_hist_broadcast_and_flatten_returns_not_implemented_on_type_error():
    """
    broadcast_and_flatten must return the NotImplemented singleton (not the
    NotImplementedError class) when ak.Array construction raises TypeError.
    Callers check `result is NotImplemented`; returning the class is truthy
    but never equal to the singleton.
    """
    from awkward._connect.hist import broadcast_and_flatten

    # Pass an object that cannot be coerced to ak.Array (raises TypeError)
    result = broadcast_and_flatten([object()])
    assert result is NotImplemented, (
        f"Expected NotImplemented singleton, got {result!r}"
    )


def test_hist_broadcast_and_flatten_works_on_valid_arrays():
    """broadcast_and_flatten returns flat numpy arrays for broadcastable input."""
    from awkward._connect.hist import broadcast_and_flatten

    a = ak.Array([1, 2, 3])
    b = ak.Array([4, 5, 6])
    result = broadcast_and_flatten([a, b])
    assert isinstance(result, tuple)
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], [1, 2, 3])
    np.testing.assert_array_equal(result[1], [4, 5, 6])
