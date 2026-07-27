# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

# Branch coverage for the descriptive-statistics changes in PR #4232:
#   * ak.var  weighted / complex-weighted two-pass, and the one-pass fallback
#     taken for a non-innermost axis over a ragged array;
#   * ak.covar / ak.corr the same fallback, plus the named-axis strip that keeps
#     the two-pass path from being rejected by the named-axis compatibility check;
#   * the SumOfSquares / SumOfPowers reducer dtype guards;
#   * the float64-accumulator bool-sum cast and its defensive branch;
#   * the typetracer sum honouring an explicit dtype.
#
# The unweighted numeric reductions here go through the awkward-cpp kernels added
# by this PR (float64-accumulator sum, sum-of-squares/powers); those cases need a
# freshly built awkward-cpp. The complex, weighted, error-path, named-axis and
# typetracer cases run without them.


# --- helpers ---------------------------------------------------------------


def _colwise(rows, fn):
    """Apply `fn` to each column of a ragged list of rows (axis=0 reduction)."""
    width = max(len(r) for r in rows)
    result = []
    for j in range(width):
        col = np.array([r[j] for r in rows if len(r) > j], dtype=np.float64)
        result.append(fn(col))
    return result


def _wvar(c, w):
    a = np.average(c, weights=w)
    return np.average((c - a) ** 2, weights=w)


def _wcov(cx, cy, w):
    ax, ay = np.average(cx, weights=w), np.average(cy, weights=w)
    return np.average((cx - ax) * (cy - ay), weights=w)


# --- ak.var: weighted / complex two-pass (axis=None) -----------------------


def test_var_weighted_two_pass():
    # Exercises the weighted sumw (`ak.sum(x*0+weight)`), the `dev = x - xmean`
    # centring, and the weighted `ak.sum(weight * dev * dev)` accumulation.
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 2.0, 3.0])
    result = ak.var(ak.Array(x), weight=ak.Array(w))
    assert result == pytest.approx(_wvar(x, w))


def test_var_complex_weighted_two_pass():
    # Complex + weight: `abs(dev)**2` then `* weight`, summed via ak.sum.
    data = np.array([1 + 2j, 3 + 4j, 5 + 1j])
    w = np.array([1.0, 2.0, 3.0])
    result = ak.var(ak.Array(data), weight=ak.Array(w))
    mu = np.average(data, weights=w)
    expected = np.average(np.abs(data - mu) ** 2, weights=w)
    assert np.imag(result) == pytest.approx(0.0)
    assert float(np.real(result)) == pytest.approx(float(expected))


# --- ak.var: one-pass fallback for a non-innermost ragged axis --------------


def test_var_fallback_weighted_ragged_axis0():
    # axis=0 over a ragged array: `x - xmean` cannot broadcast, so the weighted
    # one-pass fallback runs (promote x, sum(xp*weight, float64), sum(xp*xp*w)).
    rows = [[1, 2, 3], [4, 5, 6], [7, 8]]
    wrows = [[1, 2, 3], [4, 5, 6], [7, 8]]
    x = ak.values_astype(ak.Array(rows), np.int32)
    w = ak.values_astype(ak.Array(wrows), np.int32)
    out = ak.var(x, weight=w, axis=0)

    def col_wvar(j):
        c = np.array([r[j] for r in rows if len(r) > j], dtype=np.float64)
        wc = np.array([r[j] for r in wrows if len(r) > j], dtype=np.float64)
        return _wvar(c, wc)

    got = ak.to_list(out)
    for j, g in enumerate(got):
        assert g == pytest.approx(col_wvar(j))


def test_var_fallback_complex_ragged_axis0():
    # The complex branch of the fallback uses |x|**2 and |mean|**2, so it returns
    # NumPy's variance E[|x - mean|**2] (a real number) -- consistent with the
    # innermost complex path, not the E[x**2]-E[x]**2 pseudo-variance.
    rows = [[1 + 1j, 2 + 0j, 3 - 1j], [4 + 2j, 5 + 0j, 6 + 1j], [7 + 0j, 8 - 2j]]
    x = ak.Array(rows)
    out = ak.var(x, axis=0)

    def col_var(j):
        c = np.array([r[j] for r in rows if len(r) > j], dtype=np.complex128)
        return np.var(c)

    for j, g in enumerate(ak.to_list(out)):
        assert np.imag(g) == pytest.approx(0.0)
        assert float(np.real(g)) == pytest.approx(float(col_var(j)))


# --- ak.var: named-axis strip (two-pass must be taken, not the fallback) -----


def test_var_named_axis_matches_unnamed():
    array = ak.Array([[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]])
    named = ak.with_named_axis(array, ("x", "y"))
    # axis=None: without the named-axis strip the named array would divert to the
    # one-pass fallback and differ in the last ULP.
    assert ak.var(array, axis=None) == ak.var(named, axis=None)


# --- ak.covar: fallback + named-axis + weighted two-pass --------------------


def test_covar_fallback_ragged_axis0():
    rows_x = [[1, 2, 3], [4, 5, 6], [7, 8]]
    rows_y = [[2, 1, 4], [6, 5, 9], [8, 7]]
    x = ak.values_astype(ak.Array(rows_x), np.int32)
    y = ak.values_astype(ak.Array(rows_y), np.int32)
    out = ak.covar(x, y, axis=0)

    def col_cov(j):
        cx = np.array([r[j] for r in rows_x if len(r) > j], dtype=np.float64)
        cy = np.array([r[j] for r in rows_y if len(r) > j], dtype=np.float64)
        return np.mean((cx - cx.mean()) * (cy - cy.mean()))

    for j, g in enumerate(ak.to_list(out)):
        assert g == pytest.approx(col_cov(j))


def test_covar_fallback_weighted_ragged_axis0():
    rows_x = [[1, 2, 3], [4, 5, 6], [7, 8]]
    rows_y = [[2, 1, 4], [6, 5, 9], [8, 7]]
    rows_w = [[1, 2, 3], [4, 5, 6], [7, 8]]
    x = ak.values_astype(ak.Array(rows_x), np.int32)
    y = ak.values_astype(ak.Array(rows_y), np.int32)
    w = ak.values_astype(ak.Array(rows_w), np.int32)
    out = ak.covar(x, y, weight=w, axis=0)

    def col(j, rows):
        return np.array([r[j] for r in rows if len(r) > j], dtype=np.float64)

    for j, g in enumerate(ak.to_list(out)):
        assert g == pytest.approx(_wcov(col(j, rows_x), col(j, rows_y), col(j, rows_w)))


def test_covar_weighted_two_pass():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 6.0, 7.0])
    w = np.array([1.0, 1.0, 2.0, 2.0])
    result = ak.covar(ak.Array(x), ak.Array(y), weight=ak.Array(w))
    assert result == pytest.approx(_wcov(x, y, w))


def test_covar_named_axis_matches_unnamed():
    x = ak.Array([[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]])
    y = ak.Array([[2.0, 1.0], [4.0], [6.0, 5.0, 9.0]])
    nx = ak.with_named_axis(x, ("i", "j"))
    ny = ak.with_named_axis(y, ("i", "j"))
    assert ak.covar(x, y, axis=None) == ak.covar(nx, ny, axis=None)


# --- ak.corr: fallback (weighted + unweighted) + named-axis + weighted -------


def test_corr_fallback_unweighted_ragged_axis0():
    rows_x = [[1, 2, 3], [4, 5, 6], [7, 8]]
    rows_y = [[2, 1, 4], [6, 5, 9], [8, 7]]
    x = ak.values_astype(ak.Array(rows_x), np.int32)
    y = ak.values_astype(ak.Array(rows_y), np.int32)
    out = ak.corr(x, y, axis=0)

    def col_corr(j):
        cx = np.array([r[j] for r in rows_x if len(r) > j], dtype=np.float64)
        cy = np.array([r[j] for r in rows_y if len(r) > j], dtype=np.float64)
        return np.corrcoef(cx, cy)[0, 1]

    for j, g in enumerate(ak.to_list(out)):
        assert g == pytest.approx(col_corr(j))


def test_corr_fallback_weighted_ragged_axis0():
    rows_x = [[1, 2, 3], [4, 5, 6], [7, 8]]
    rows_y = [[2, 1, 4], [6, 5, 9], [8, 7]]
    rows_w = [[1, 2, 3], [4, 5, 6], [7, 8]]
    x = ak.values_astype(ak.Array(rows_x), np.int32)
    y = ak.values_astype(ak.Array(rows_y), np.int32)
    w = ak.values_astype(ak.Array(rows_w), np.int32)
    out = ak.corr(x, y, weight=w, axis=0)

    def col(j, rows):
        return np.array([r[j] for r in rows if len(r) > j], dtype=np.float64)

    for j, g in enumerate(ak.to_list(out)):
        cx, cy, cw = col(j, rows_x), col(j, rows_y), col(j, rows_w)
        expected = _wcov(cx, cy, cw) / np.sqrt(_wvar(cx, cw) * _wvar(cy, cw))
        assert g == pytest.approx(expected)


def test_corr_weighted_two_pass():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 6.0, 7.0])
    w = np.array([1.0, 1.0, 2.0, 2.0])
    result = ak.corr(ak.Array(x), ak.Array(y), weight=ak.Array(w))
    expected = _wcov(x, y, w) / np.sqrt(_wvar(x, w) * _wvar(y, w))
    assert result == pytest.approx(expected)


def test_corr_named_axis_matches_unnamed():
    x = ak.Array([[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]])
    y = ak.Array([[2.0, 1.0], [4.0], [6.0, 5.0, 9.0]])
    nx = ak.with_named_axis(x, ("i", "j"))
    ny = ak.with_named_axis(y, ("i", "j"))
    assert ak.corr(x, y, axis=None) == ak.corr(nx, ny, axis=None)


# --- reducer dtype guards ---------------------------------------------------


def test_sumofsquares_rejects_complex():
    with pytest.raises(TypeError, match="sum-of-squares"):
        ak.operations.ak_sumofsquares._impl(
            ak.Array([1 + 2j, 3 + 4j]), None, False, False, True, None, None
        )


def test_sumofsquares_rejects_datetime():
    data = ak.Array(np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"))
    with pytest.raises(ValueError, match="sum-of-squares"):
        ak.operations.ak_sumofsquares._impl(data, None, False, False, True, None, None)


def test_sumofpowers_rejects_complex():
    with pytest.raises(TypeError, match="sum-of-powers"):
        ak.operations.ak_sumofpowers._impl(
            ak.Array([1 + 2j, 3 + 4j]), 3, None, False, False, True, None, None
        )


def test_sumofpowers_rejects_datetime():
    data = ak.Array(np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"))
    with pytest.raises(ValueError, match="sum-of-powers"):
        ak.operations.ak_sumofpowers._impl(
            data, 3, None, False, False, True, None, None
        )


# --- float64-accumulator bool sum (via ak.mean) -----------------------------


def test_mean_bool_float64_accumulator():
    # A bool array through the float64-accumulator sum: the bool kernel produces
    # an integer per-segment count which is then cast to the requested float64.
    array = ak.Array([[True, False, True], [False, True]])
    out = ak.mean(array, axis=1)
    assert ak.to_list(out) == [pytest.approx(2 / 3), pytest.approx(1 / 2)]


def test_sum_bool_forced_dtype_unsupported_result(monkeypatch):
    # Defensive branch: if the bool accumulator's promoted result dtype is
    # neither 32- nor 64-bit integer, the reducer raises NotImplementedError.
    monkeypatch.setattr(
        ak._reducers.Sum,
        "_promote_integer_rank",
        lambda self, dt: np.dtype("float16").type,
    )
    array = ak.Array([[True, False, True], [False]])
    with pytest.raises(NotImplementedError):
        ak.operations.ak_sum._impl(
            array, 1, False, False, True, None, None, dtype=np.float64
        )


# --- Sum.apply numeric segmented path (plain / forced-accumulator / complex) -


def test_sum_numeric_segmented_plain():
    # Numeric, non-complex, no forced dtype (use_forced=False): the segmented
    # awkward_reduce_sum runs, then the result is viewed back to the promoted
    # integer rank (int32 -> int64).
    array = ak.values_astype(ak.Array([[1, 2, 3], [], [4, 5]]), np.int32)
    out = ak.sum(array, axis=1)
    assert ak.to_list(out) == [6, 0, 9]
    # Result is viewed back to the promoted integer rank (platform intp: int64 on
    # 64-bit POSIX, int32 on Windows) -- assert the kind, not the exact width.
    assert ak.to_numpy(out).dtype.kind == "i"


def test_sum_numeric_segmented_forced_float64_accumulator():
    # Numeric with a forced float64 accumulator (use_forced=True): the same
    # segmented kernel sums directly into float64, returned without a view-back.
    array = ak.values_astype(ak.Array([[1, 2, 3], [], [4, 5]]), np.int32)
    out = ak.operations.ak_sum._impl(
        array, 1, False, False, True, None, None, dtype=np.float64
    )
    assert ak.to_list(out) == [6.0, 0.0, 9.0]
    assert ak.to_numpy(out).dtype == np.dtype(np.float64)


def test_sum_complex_segmented():
    # Complex numeric (is_complex=True): the awkward_reduce_sum_complex branch.
    array = ak.Array([[1 + 2j, 3 + 0j], [], [4 - 1j]])
    out = ak.sum(array, axis=1)
    assert ak.to_list(out) == [4 + 2j, 0j, 4 - 1j]


# --- AxisNoneSum.apply (axis=None routes through nplike.sum) ------------------


def test_sum_axis_none_plain():
    # axis=None, no forced dtype: reduce via nplike.sum (else branch), reshaped
    # to length 1 and wrapped -- exercises the AxisNoneSum return path.
    array = ak.values_astype(ak.Array([[1, 2, 3], [], [4, 5]]), np.int32)
    out = ak.sum(array, axis=None)
    assert out == 15
    assert np.asarray(out).dtype.kind == "i"


def test_sum_axis_none_forced_float64_accumulator():
    # axis=None with a forced float64 accumulator (self._dtype set, not complex):
    # nplike.sum is called with dtype=float64.
    array = ak.values_astype(ak.Array([[1, 2, 3], [], [4, 5]]), np.int32)
    out = ak.operations.ak_sum._impl(
        array, None, False, False, True, None, None, dtype=np.float64
    )
    assert out == 15.0
    assert np.asarray(out).dtype == np.dtype(np.float64)


def test_sum_axis_none_complex():
    # axis=None complex: dtype.kind == "c" forces the plain (no-dtype) reduce.
    array = ak.Array([[1 + 2j, 3 + 0j], [], [4 - 1j]])
    out = ak.sum(array, axis=None)
    assert out == 8 + 1j


# --- moment: unweighted complex uses sum(x**n), not the float64 reducer ------


def test_moment_complex_unweighted():
    # The sum-of-squares/powers reducers are float64-only; complex input must
    # route through sum(x**n) (both n==2 and other n), matching NumPy -- not
    # raise TypeError from the reducer.
    data = np.array([1 + 2j, 3 + 4j, 5 + 1j])
    for n in (2, 3):
        assert ak.moment(ak.Array(data), n) == pytest.approx(np.mean(data**n))


def test_moment_complex_unweighted_jagged():
    array = ak.Array([[1 + 2j, 3 + 4j], [5 + 1j]])
    out = ak.moment(array, 2, axis=-1)
    assert out[0] == pytest.approx(np.mean(np.array([1 + 2j, 3 + 4j]) ** 2))
    assert out[1] == pytest.approx((5 + 1j) ** 2)


# --- mean of timedelta keeps its dtype (not float64 tick counts) -------------


def _timedelta_jagged():
    flat = np.array([1, 2, 3, 4, 5], dtype="timedelta64[s]")
    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 3, 5])),
            ak.contents.NumpyArray(flat),
        )
    )


def test_mean_timedelta_concrete_axis_keeps_timedelta():
    # The forced float64 accumulator must NOT apply to timedelta (kind "m"):
    # an explicit axis must return timedelta64, not raw float64 tick counts.
    arr = _timedelta_jagged()
    out = ak.mean(arr, axis=1, mask_identity=True)
    assert "timedelta64" in str(out.type)
    assert ak.to_list(out) == [np.timedelta64(2, "s"), np.timedelta64(4, "s")]


def test_mean_timedelta_axis_none_keeps_timedelta():
    arr = _timedelta_jagged()
    out = ak.mean(arr, axis=None)
    assert isinstance(out, np.timedelta64)
    assert out == np.timedelta64(3, "s")


# --- typetracer honours an explicit dtype -----------------------------------


def test_typetracer_sum_explicit_dtype():
    # ak.var on a typetracer routes through the float64-accumulator sum, whose
    # explicit dtype must be honoured by the typetracer nplike (not re-derived
    # from the integer input dtype).
    base = ak.values_astype(ak.Array([[1, 2, 3], [4, 5]]), np.int32)
    tt = ak.to_backend(base, "typetracer")
    assert str(ak.var(tt, axis=-1).type) == "2 * float64"
    assert str(ak.mean(tt, axis=-1).type) == "2 * float64"
