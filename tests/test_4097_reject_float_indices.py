# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

import awkward as ak


def test_float_index_raises():
    arr = ak.Array([10, 20, 30])
    with pytest.raises(TypeError):
        arr[1.5]


def test_decimal_index_raises():
    # Decimal has __int__ but not __index__; the fix uses __index__ protocol
    arr = ak.Array([10, 20, 30])
    with pytest.raises(TypeError):
        arr[Decimal("1")]


def test_custom_index_object_works():
    class MyIndex:
        def __index__(self):
            return 1

    arr = ak.Array([10, 20, 30])
    assert arr[MyIndex()] == 20


def test_int_index_works():
    arr = ak.Array([10, 20, 30])
    assert arr[0] == 10
    assert arr[np.int64(2)] == 30
    assert arr[-1] == 30


def test_axis_float_raises():
    arr2d = ak.Array([[1, 2], [3, 4]])
    with pytest.raises((TypeError, ValueError)):
        ak.sum(arr2d, axis=1.5)


def test_axis_int_works():
    arr2d = ak.Array([[1, 2], [3, 4]])
    assert ak.sum(arr2d, axis=1).tolist() == [3, 7]
    assert ak.sum(arr2d, axis=np.int64(1)).tolist() == [3, 7]
