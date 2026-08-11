# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np
import pytest

import awkward as ak

# ak.var/ak.std on the numpy backend, unweighted, at the *innermost* axis use the
# fused centered sum-of-squares reducer: Sigma (x - mean)**2 per segment in one
# pass -- no deviation buffer and no mean back-broadcast. Deviations are formed in
# float64 inside the kernel, so it is overflow-safe and cancellation-stable. Only
# the innermost axis is fused: for a non-innermost axis the reduce transposes the
# content, which would misalign the per-bin means, so those use the two-pass path
# (still asserted correct here). Fused cases need the
# awkward_reduce_centered_sumofsquares kernel (built with awkward-cpp).


DTYPES = [
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "float32",
    "float64",
]


@pytest.mark.parametrize("dtype", DTYPES)
def test_var_std_axis_last_matches_numpy(dtype):
    rows = [[3, 1, 2, 4], [5, 6], [7]]
    arr = ak.values_astype(ak.Array(rows), dtype)
    for op, npop in ((ak.var, np.var), (ak.std, np.std)):
        got = ak.to_list(op(arr, axis=-1))
        for g, r in zip(got, rows, strict=True):
            assert g == pytest.approx(npop(np.array(r, dtype=np.float64)))


def test_var_int32_overflow_grouped_axis():
    # (value - mean)**2 with int32 values whose squares overflow int32: the
    # kernel centres in float64, so the result is correct.
    arr = ak.values_astype(
        ak.Array([[100000, 200000, 300000], [400000, 500000]]), np.int32
    )
    got = ak.to_list(ak.var(arr, axis=-1))
    assert got[0] == pytest.approx(np.var([1e5, 2e5, 3e5]))
    assert got[1] == pytest.approx(np.var([4e5, 5e5]))


def test_var_float32_cancellation_grouped_axis():
    # Large-mean float32: one-pass E[x**2]-E[x]**2 cancels; the centred kernel
    # gives 2/3.
    arr = ak.values_astype(ak.Array([[1e7, 1e7 + 1, 1e7 + 2]]), np.float32)
    assert ak.var(arr, axis=-1)[0] == pytest.approx(2.0 / 3.0, rel=1e-9)


def _colwise(rows, fn):
    width = max(len(r) for r in rows)
    return [
        fn(np.array([r[j] for r in rows if len(r) > j], dtype=np.float64))
        for j in range(width)
    ]


def test_var_non_innermost_ragged_axis0():
    # axis=0 over a ragged array is NOT innermost, so it does not use the fused
    # kernel (the reduce transposes the content there, which would misalign the
    # per-bin means); it uses the two-pass/one-pass path. Result must still match.
    rows = [[1, 2, 3], [4, 5, 6], [7, 8]]
    arr = ak.values_astype(ak.Array(rows), np.int32)
    got = ak.to_list(ak.var(arr, axis=0))
    for g, e in zip(got, _colwise(rows, np.var), strict=True):
        assert g == pytest.approx(e)


def test_var_three_deep_axis0_regression():
    # Regression for a means-misalignment bug: the fused kernel must NOT be used
    # for axis=0 on a deeply-ragged array (see test_2918). Values must be right.
    data = ak.Array(
        [[[0, 1.1, 2.2], []], [], [[3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]]
    )
    assert ak.var(data, axis=0).to_list() == [
        [pytest.approx(2.7225), pytest.approx(2.7225), pytest.approx(0.0)],
        [pytest.approx(0.0)],
        [
            pytest.approx(0.0),
            pytest.approx(0.0),
            pytest.approx(0.0),
            pytest.approx(0.0),
        ],
    ]


def test_var_three_deep_middle_axis():
    arr = ak.Array([[[1.0, 2], [3, 4]], [[5.0, 6], [7, 8]]])
    # axis=1 combines the two middle lists within each outer element.
    assert ak.to_list(ak.var(arr, axis=1)) == [
        [pytest.approx(1.0), pytest.approx(1.0)],
        [pytest.approx(1.0), pytest.approx(1.0)],
    ]
    # axis=2 is the innermost.
    assert ak.to_list(ak.var(arr, axis=2)) == [
        [pytest.approx(0.25), pytest.approx(0.25)],
        [pytest.approx(0.25), pytest.approx(0.25)],
    ]


def test_var_rectangular_axes():
    M = np.arange(12.0).reshape(3, 4)
    arr = ak.Array(M)
    assert ak.to_list(ak.var(arr, axis=0)) == [
        pytest.approx(v) for v in np.var(M, axis=0)
    ]
    assert ak.to_list(ak.var(arr, axis=1)) == [
        pytest.approx(v) for v in np.var(M, axis=1)
    ]


def test_var_empty_sublist_is_masked():
    arr = ak.Array([[1.0, 2, 3], [], [6.0, 7]])
    out = ak.var(arr, axis=-1, mask_identity=True)
    assert out[0] == pytest.approx(np.var([1, 2, 3]))
    assert out[1] is None
    assert out[2] == pytest.approx(np.var([6, 7]))


def test_var_ddof_grouped_axis():
    rows = [[3.0, 1, 2, 4], [5.0, 6, 7]]
    arr = ak.Array(rows)
    got = ak.to_list(ak.var(arr, axis=-1, ddof=1))
    for g, r in zip(got, rows, strict=True):
        assert g == pytest.approx(np.var(np.array(r), ddof=1))


def test_var_equals_covar_self_grouped_axis():
    # var (fused) and covar(x, x) (two-pass) are the same quantity and must agree,
    # even with a large mean where a one-pass form would cancel.
    x = ak.Array([[1e9, 1e9 + 1, 1e9 + 2], [5.0, 7, 9, 11]])
    v = ak.to_list(ak.var(x, axis=-1))
    c = ak.to_list(ak.covar(x, x, axis=-1))
    for a, b in zip(v, c, strict=True):
        assert a == pytest.approx(b)


def test_var_all_equal_not_negative():
    arr = ak.values_astype(
        ak.Array([np.full(100000, 1000.0, dtype=np.float32)]), np.float32
    )
    assert ak.var(arr, axis=-1)[0] == pytest.approx(0.0, abs=1e-6)
    assert ak.std(arr, axis=-1)[0] == pytest.approx(0.0, abs=1e-6)
