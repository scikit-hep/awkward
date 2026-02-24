# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    try:
        cp.cuda.Device().synchronize()
    except cp.cuda.runtime.CUDARuntimeError as e:
        print("GPU error during sync:", e)
    cp._default_memory_pool.free_all_blocks()


@pytest.mark.parametrize("axis", [None, 0, -1])
def test_sum_axis_none_non_empty(axis):
    array = ak.Array([1, 2, 3, 4, 5], backend="cuda")
    assert ak.sum(array, axis=axis) == 15

    array = ak.Array([1.0, 2.0, 3.0], backend="cuda")
    assert ak.sum(array, axis=axis) == 6.0


@pytest.mark.parametrize("axis", [None, 0, -1])
def test_sum_axis_none_empty(axis):
    array = ak.Array(np.array([], dtype=np.int64), backend="cuda")
    assert ak.sum(array, axis=axis) == 0

    array = ak.Array(np.array([], dtype=np.float64), backend="cuda")
    assert ak.sum(array, axis=axis) == 0.0


@pytest.mark.parametrize("axis", [None, 0, -1])
def test_min_axis_none_non_empty(axis):
    int_array = ak.Array([3, 1, 4, 1, 5, 9, 2, 6], backend="cuda")
    float_array = ak.Array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], backend="cuda")

    # basic (default mask_identity=True, non-empty so not masked)
    assert ak.min(int_array, axis=axis) == 1
    assert ak.min(float_array, axis=axis) == 1.0

    # mask_identity=False
    assert ak.min(int_array, axis=axis, mask_identity=False) == 1
    assert ak.min(float_array, axis=axis, mask_identity=False) == 1.0

    # mask_identity=True (non-empty, so result is not masked)
    assert ak.min(int_array, axis=axis, mask_identity=True) == 1
    assert ak.min(float_array, axis=axis, mask_identity=True) == 1.0

    # initial lower than array minimum → initial wins
    assert ak.min(int_array, axis=axis, initial=0, mask_identity=False) == 0
    assert ak.min(float_array, axis=axis, initial=0.0, mask_identity=False) == 0.0

    # initial higher than array minimum → array wins
    assert ak.min(int_array, axis=axis, initial=2, mask_identity=False) == 1
    assert ak.min(float_array, axis=axis, initial=2.0, mask_identity=False) == 1.0

    # initial with mask_identity=True (non-empty, so result is not masked)
    assert ak.min(int_array, axis=axis, initial=0, mask_identity=True) == 0
    assert ak.min(float_array, axis=axis, initial=0.0, mask_identity=True) == 0.0


@pytest.mark.parametrize("axis", [None, 0, -1])
def test_min_axis_none_empty(axis):
    int_array = ak.Array(np.array([], dtype=np.int64), backend="cuda")
    float_array = ak.Array(np.array([], dtype=np.float64), backend="cuda")

    # default (mask_identity=True) → None
    assert ak.min(int_array, axis=axis) is None
    assert ak.min(float_array, axis=axis) is None

    # mask_identity=False → identity element
    assert ak.min(int_array, axis=axis, mask_identity=False) == np.iinfo(np.int64).max
    assert ak.min(float_array, axis=axis, mask_identity=False) == np.inf

    # initial with mask_identity=False → initial
    assert ak.min(int_array, axis=axis, initial=5, mask_identity=False) == 5
    assert ak.min(float_array, axis=axis, initial=5.0, mask_identity=False) == 5.0

    # initial with mask_identity=True (default) → still None
    assert ak.min(int_array, axis=axis, initial=5) is None
    assert ak.min(float_array, axis=axis, initial=5.0) is None


@pytest.mark.parametrize("axis", [None, 0, -1])
def test_max_axis_none_non_empty(axis):
    int_array = ak.Array([3, 1, 4, 1, 5, 9, 2, 6], backend="cuda")
    float_array = ak.Array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], backend="cuda")

    # basic (default mask_identity=True, non-empty so not masked)
    assert ak.max(int_array, axis=axis) == 9
    assert ak.max(float_array, axis=axis) == 9.0

    # mask_identity=False
    assert ak.max(int_array, axis=axis, mask_identity=False) == 9
    assert ak.max(float_array, axis=axis, mask_identity=False) == 9.0

    # mask_identity=True (non-empty, so result is not masked)
    assert ak.max(int_array, axis=axis, mask_identity=True) == 9
    assert ak.max(float_array, axis=axis, mask_identity=True) == 9.0

    # initial higher than array maximum → initial wins
    assert ak.max(int_array, axis=axis, initial=10, mask_identity=False) == 10
    assert ak.max(float_array, axis=axis, initial=10.0, mask_identity=False) == 10.0

    # initial lower than array maximum → array wins
    assert ak.max(int_array, axis=axis, initial=2, mask_identity=False) == 9
    assert ak.max(float_array, axis=axis, initial=2.0, mask_identity=False) == 9.0

    # initial with mask_identity=True (non-empty, so result is not masked)
    assert ak.max(int_array, axis=axis, initial=10, mask_identity=True) == 10
    assert ak.max(float_array, axis=axis, initial=10.0, mask_identity=True) == 10.0


@pytest.mark.parametrize("axis", [None, 0, -1])
def test_max_axis_none_empty(axis):
    int_array = ak.Array(np.array([], dtype=np.int64), backend="cuda")
    float_array = ak.Array(np.array([], dtype=np.float64), backend="cuda")

    # default (mask_identity=True) → None
    assert ak.max(int_array, axis=axis) is None
    assert ak.max(float_array, axis=axis) is None

    # mask_identity=False → identity element
    assert ak.max(int_array, axis=axis, mask_identity=False) == np.iinfo(np.int64).min
    assert ak.max(float_array, axis=axis, mask_identity=False) == -np.inf

    # initial with mask_identity=False → initial
    assert ak.max(int_array, axis=axis, initial=5, mask_identity=False) == 5
    assert ak.max(float_array, axis=axis, initial=5.0, mask_identity=False) == 5.0

    # initial with mask_identity=True (default) → still None
    assert ak.max(int_array, axis=axis, initial=5) is None
    assert ak.max(float_array, axis=axis, initial=5.0) is None
