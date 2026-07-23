# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def _has_float64_integer_kernel():
    # The segmented (axis != None) float64/integer accumulator needs the
    # awkward_reduce_sum_float64_<int>_64 kernels; skip if awkward-cpp predates
    # them.
    try:
        ak.sum(ak.Array([[1, 2], [3]]), axis=-1, dtype=np.float64)
    except KeyError:
        return False
    return True


needs_kernel = pytest.mark.skipif(
    not _has_float64_integer_kernel(),
    reason="requires awkward-cpp built with float64/integer reduce_sum kernels",
)


def test_sum_dtype_none_unchanged():
    a = ak.Array([[1, 2, 3], [], [4, 5]])
    assert ak.sum(a, axis=-1).to_list() == [6, 0, 9]
    assert str(ak.sum(a, axis=-1).type) == "3 * int64"
    assert ak.sum(a) == 15


def test_sum_axis_none_float64_accumulator():
    # int32 total (6e9) exceeds int32 but the float64 accumulator is exact here.
    data = np.array([2_000_000_000, 2_000_000_000, 2_000_000_000], dtype=np.int32)
    out = ak.sum(ak.Array(data), axis=None, dtype=np.float64)
    assert out == pytest.approx(np.sum(data, dtype=np.float64))
    assert np.asarray(out).dtype == np.dtype(np.float64)


def test_sum_axis_none_bool_float64():
    out = ak.sum(ak.Array(np.array([True, False, True])), axis=None, dtype=np.float64)
    assert out == 2.0


def test_sum_axis_none_float_input_unchanged():
    data = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    assert ak.sum(ak.Array(data), axis=None) == pytest.approx(np.sum(data))


@needs_kernel
def test_sum_segmented_float64_accumulator():
    a = ak.Array([[1, 2, 3], [], [4, 5]])
    out = ak.sum(a, axis=-1, dtype=np.float64)
    assert out.to_list() == [6.0, 0.0, 9.0]
    assert str(out.type) == "3 * float64"


@needs_kernel
def test_sum_segmented_no_overflow():
    # int32 squares would overflow, but here we sum large int32 values into
    # float64 directly (no promoted input copy).
    a = ak.Array([[2_000_000_000, 2_000_000_000], [2_000_000_000]])
    out = ak.sum(a, axis=-1, dtype=np.float64)
    assert out.to_list() == [4_000_000_000.0, 2_000_000_000.0]
