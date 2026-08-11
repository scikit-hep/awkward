# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np
import pytest

import awkward as ak

# ak.linear_fit formed x**2 and x*y at the input dtype before summing, so integer
# squares/products overflowed (e.g. int32 100000**2 wraps) and corrupted the fit
# (delta = sumw*sumwxx - sumwx*sumwx also overflowed). x and y are now promoted to
# float64 before the products, so the fit is accumulated in float64. Complex and
# float inputs are unchanged.


def _fit(x, y, weight=None, axis=None):
    return ak.linear_fit(
        ak.Array(x),
        ak.Array(y),
        weight=None if weight is None else ak.Array(weight),
        axis=axis,
    )


def test_linear_fit_int32_overflow():
    # 100000**2 overflows int32; y = 2x + 1.
    x = np.array([100000, 200000, 300000, 400000], dtype=np.int32)
    y = (2 * x + 1).astype(np.int32)
    fit = _fit(x, y)
    assert fit["slope"] == pytest.approx(2.0)
    assert fit["intercept"] == pytest.approx(1.0)


def test_linear_fit_int64_overflow():
    # The elementwise x*x products must exceed int64 before promotion for this test
    # to exercise the fix: x ~ 3-5e9 so the 4e9 and 5e9 squares (1.6e19, 2.5e19) wrap
    # int64 (max ~9.2e18), and the delta-stage products wrap too. On main these inputs
    # give slope -0.0745; promoted they give 3.0.
    # intercept magnitude is kept large (-7e6) so its float64 rounding
    # (-7000000.000018) stays within pytest.approx's default relative tolerance.
    x = np.array([3_000_000_000, 4_000_000_000, 5_000_000_000], dtype=np.int64)
    y = 3 * x - 7_000_000
    fit = _fit(x, y)
    assert fit["slope"] == pytest.approx(3.0)
    assert fit["intercept"] == pytest.approx(-7_000_000.0)


def test_linear_fit_uint_and_negative():
    for dtype in (np.uint32, np.int32):
        x = np.array([100000, 200000, 300000], dtype=dtype)
        y = (2 * x.astype(np.int64) + 5).astype(dtype)
        fit = _fit(x, y)
        assert fit["slope"] == pytest.approx(2.0)
        assert fit["intercept"] == pytest.approx(5.0)

    # signed input with genuinely negative x (the uint loop above cannot).
    x = np.array([-300000, -100000, 200000, 400000], dtype=np.int32)
    y = (2 * x + 5).astype(np.int32)
    fit = _fit(x, y)
    assert fit["slope"] == pytest.approx(2.0)
    assert fit["intercept"] == pytest.approx(5.0)


def test_linear_fit_mixed_integer_complex_overflow():
    # int x and complex y: promoting each array independently keeps x*x in float64
    # (int32 100000**2 would otherwise wrap) while y stays complex. Without the
    # promotion the int32 x*x products wrap, giving slope (-0.4059-0j).
    xi = np.array([100000, 200000, 300000, 400000], dtype=np.int32)
    x = ak.Array(xi)
    y = ak.Array((2 * xi + 1).astype(np.complex128))
    fit = ak.linear_fit(x, y)
    assert fit["slope"] == pytest.approx(2 + 0j)
    assert fit["intercept"] == pytest.approx(1 + 0j)


def test_linear_fit_weighted_int32_overflow():
    x = np.array([100000, 200000, 300000], dtype=np.int32)
    y = (2 * x + 1).astype(np.int32)
    w = np.array([1.0, 2.0, 3.0])
    fit = _fit(x, y, weight=w)
    assert fit["slope"] == pytest.approx(2.0)
    assert fit["intercept"] == pytest.approx(1.0)


def test_linear_fit_matches_numpy_polyfit():
    rng = np.random.default_rng(0)
    x = rng.integers(50000, 300000, size=25).astype(np.int32)
    y = (3 * x.astype(np.int64) - 11).astype(np.int32)
    fit = _fit(x, y)
    sl, ic = np.polyfit(x.astype(np.float64), y.astype(np.float64), 1)
    assert fit["slope"] == pytest.approx(sl)
    assert fit["intercept"] == pytest.approx(ic)


def test_linear_fit_float64_unchanged():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 1.0
    fit = _fit(x, y)
    assert fit["slope"] == pytest.approx(2.0)
    assert fit["intercept"] == pytest.approx(1.0)


def test_linear_fit_complex_not_regressed():
    # Complex inputs pass through unchanged (promote_integral_to_float64 is a
    # no-op for complex leaves). Points lie exactly on y = (2+1j)*x + (1-1j), so
    # the least-squares fit recovers those coefficients exactly.
    x = [1 + 0j, 2 + 0j, 3 + 0j]
    y = [(2 + 1j) * xi + (1 - 1j) for xi in x]
    fit = _fit(x, y)
    assert fit["slope"] == pytest.approx(2 + 1j)
    assert fit["intercept"] == pytest.approx(1 - 1j)


def test_linear_fit_jagged_axis1_overflow():
    x = ak.values_astype(
        ak.Array([[100000, 200000, 300000], [10, 20, 30, 40]]), np.int32
    )
    y = ak.values_astype(
        ak.Array([[200001, 400001, 600001], [21, 41, 61, 81]]), np.int32
    )
    fit = ak.linear_fit(x, y, axis=1)
    assert ak.to_list(fit["slope"]) == [pytest.approx(2.0), pytest.approx(2.0)]
    assert ak.to_list(fit["intercept"]) == [pytest.approx(1.0), pytest.approx(1.0)]
