# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

# ak.var/ak.std/ak.moment are two-pass (centre on the float64 mean, then sum the
# squared deviations via the fused sum-of-squares reducer). This is numerically
# stable (no E[x**2]-E[x]**2 cancellation) and overflow-safe (deviations are
# float64). Unweighted numeric cases need the awkward_reduce_sumofsquares kernel
# (built with awkward-cpp); complex and weighted cases run through ak.sum.


# --- overflow (the original motivation) ------------------------------------


def test_var_int32_overflow():
    data = np.array([100000, 200000, 300000], dtype=np.int32)
    assert ak.var(ak.Array(data)) == pytest.approx(np.var(data.astype(np.float64)))
    assert ak.std(ak.Array(data)) == pytest.approx(np.std(data.astype(np.float64)))


def test_var_int64_overflow():
    # Deviations overflow int64 in the old form; two-pass in float64 is fine.
    data = np.array([3_000_000_000, 4_000_000_000, 5_000_000_000], dtype=np.int64)
    assert ak.var(ak.Array(data)) == pytest.approx(np.var(data.astype(np.float64)))
    assert not np.isnan(ak.std(ak.Array(data)))


def test_var_uint_and_negative():
    for data in (
        np.array([100000, 200000, 300000], dtype=np.uint32),
        np.array([-100000, 200000, -300000], dtype=np.int32),
    ):
        assert ak.var(ak.Array(data)) == pytest.approx(np.var(data.astype(np.float64)))


# --- cancellation: what the one-pass form got wrong ------------------------


def test_var_float32_cancellation():
    # One-pass E[x**2]-E[x]**2 returns ~0.672 (or nan) here; two-pass gives 2/3.
    data = np.array([1e7, 1e7 + 1, 1e7 + 2], dtype=np.float32)
    assert ak.var(ak.Array(data)) == pytest.approx(2.0 / 3.0, rel=1e-9)


def test_var_all_equal_not_nan():
    # A large run of identical float32 values: one-pass returned a negative
    # variance / nan std; two-pass returns exactly 0.
    data = np.full(1_000_000, 1000.0, dtype=np.float32)
    assert ak.var(ak.Array(data)) == pytest.approx(0.0, abs=1e-6)
    assert ak.std(ak.Array(data)) == pytest.approx(0.0, abs=1e-6)


def test_var_matches_covar_self():
    # var(x) and covar(x, x) are the same quantity and must agree even when the
    # mean is large relative to the spread (where one-pass cancels).
    x = ak.Array([1e9, 1e9 + 1, 1e9 + 2])
    assert ak.var(x) == pytest.approx(ak.covar(x, x))
    assert ak.var(x) == pytest.approx(2.0 / 3.0)


# --- complex (two-pass, real result -- runs via ak.sum) --------------------


def test_var_complex_is_real():
    # NumPy defines variance of complex data as E[|x - mean|**2], a real number.
    data = np.array([1 + 2j, 3 + 4j, 5 + 1j])
    result = ak.var(ak.Array(data))
    assert np.imag(result) == pytest.approx(0.0)
    assert float(np.real(result)) == pytest.approx(float(np.var(data)))


# --- weighted (two-pass, runs via ak.sum) ----------------------------------


def _np_weighted_var(x, w):
    x, w = x.astype(np.float64), w.astype(np.float64)
    avg = np.average(x, weights=w)
    return np.average((x - avg) ** 2, weights=w)


def test_weighted_var_and_std_overflow():
    x = np.array([100000, 200000, 300000], dtype=np.int32)
    w = np.array([1, 2, 3], dtype=np.int32)
    assert ak.var(ak.Array(x), weight=ak.Array(w)) == pytest.approx(
        _np_weighted_var(x, w)
    )
    assert ak.std(ak.Array(x), weight=ak.Array(w)) == pytest.approx(
        np.sqrt(_np_weighted_var(x, w))
    )


# --- structure: jagged, keepdims, masking, nan-variants --------------------


def test_var_jagged_keepdims_mask():
    array = ak.values_astype(
        ak.Array([[100000, 200000, 300000], [], [400000, 500000]]), np.int32
    )
    out = ak.var(array, axis=-1, mask_identity=True)
    assert out[0] == pytest.approx(np.var([1e5, 2e5, 3e5]))
    assert out[1] is None
    assert out[2] == pytest.approx(np.var([4e5, 5e5]))

    kd = ak.var(
        ak.values_astype(ak.Array([[1, 2, 3]]), np.int32), axis=-1, keepdims=True
    )
    assert kd.to_list() == [[pytest.approx(np.var([1.0, 2.0, 3.0]))]]


def test_nanvar_integer():
    data = np.array([100000, 200000, 300000], dtype=np.int32)
    assert ak.nanvar(ak.Array(data)) == pytest.approx(np.var(data.astype(np.float64)))
    assert ak.nanstd(ak.Array(data)) == pytest.approx(np.std(data.astype(np.float64)))


def test_var_typetracer_is_float64():
    base = ak.values_astype(ak.Array([[1, 2, 3], [4, 5]]), np.int32)
    tt = ak.to_backend(base, "typetracer")
    assert str(ak.var(tt, axis=-1).type) == "2 * float64"


def test_var_float64_unchanged():
    data = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    assert ak.var(ak.Array(data)) == pytest.approx(np.var(data))
