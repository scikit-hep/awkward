# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Tests for PR #4097: reject non-integer float indices instead of truncating.

Before this fix, ak.Array([10, 20, 30])[1.5] would silently truncate 1.5 to 1
and return 20. Now it raises TypeError, matching NumPy behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_float_scalar_index_raises():
    """Plain Python float index should raise TypeError, not silently truncate."""
    arr = ak.Array([10, 20, 30])
    with pytest.raises(TypeError):
        arr[1.5]


def test_float_exact_integer_value_raises():
    """Even a float that equals an integer (1.0) should raise, not silently truncate."""
    arr = ak.Array([10, 20, 30])
    with pytest.raises(TypeError):
        arr[1.0]


def test_np_float64_index_raises():
    """np.float64 index was already rejected; confirm it still raises."""
    arr = ak.Array([10, 20, 30])
    with pytest.raises(TypeError):
        arr[np.float64(1.5)]


def test_int_index_works():
    """Plain int index must still work."""
    arr = ak.Array([10, 20, 30])
    assert arr[0] == 10
    assert arr[1] == 20
    assert arr[2] == 30
    assert arr[-1] == 30


def test_np_int64_index_works():
    """np.int64 index must still work."""
    arr = ak.Array([10, 20, 30])
    assert arr[np.int64(0)] == 10
    assert arr[np.int64(2)] == 30


def test_np_intp_index_works():
    """np.intp index must still work."""
    arr = ak.Array([10, 20, 30])
    assert arr[np.intp(1)] == 20


def test_custom_index_object_works():
    """Object implementing __index__ must work as an index."""

    class MyIndex:
        def __index__(self):
            return 1

    arr = ak.Array([10, 20, 30])
    assert arr[MyIndex()] == 20


def test_axis_float_raises():
    """axis=1.5 should raise rather than silently truncate to axis=1."""
    arr2d = ak.Array([[1, 2], [3, 4]])
    with pytest.raises((TypeError, ValueError)):
        ak.sum(arr2d, axis=1.5)


def test_axis_int_works():
    """axis=1 must still work correctly."""
    arr2d = ak.Array([[1, 2], [3, 4]])
    result = ak.sum(arr2d, axis=1)
    assert result.tolist() == [3, 7]


def test_axis_np_int64_works():
    """axis=np.int64(1) must still work correctly."""
    arr2d = ak.Array([[1, 2], [3, 4]])
    result = ak.sum(arr2d, axis=np.int64(1))
    assert result.tolist() == [3, 7]


def test_float_index_nested():
    """Float index into nested array should raise."""
    arr = ak.Array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(TypeError):
        arr[0.5]


def test_decimal_index_raises():
    """Decimal has __int__ but not __index__; should raise."""
    from decimal import Decimal

    arr = ak.Array([10, 20, 30])
    with pytest.raises(TypeError):
        arr[Decimal("1")]
