# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

# ak.var/ak.std accumulate sum(x**2) in float64 directly from the input (via the
# awkward_reduce_sumofsquares kernels), so integer and float32 squares neither
# overflow nor lose precision and no x*x buffer is allocated.


def test_var_int32_overflow():
    # 100000**2 = 1e10 overflows int32; the old sum(x*x) path wraps.
    data = np.array([100000, 200000, 300000], dtype=np.int32)
    assert ak.var(ak.Array(data)) == pytest.approx(np.var(data.astype(np.float64)))
    assert ak.std(ak.Array(data)) == pytest.approx(np.std(data.astype(np.float64)))


def test_var_int64_overflow():
    # squares overflow int64 (the accumulator), so on main the variance goes
    # negative and ak.std returns nan (issue #3525).
    data = np.array([3_000_000_000, 4_000_000_000, 5_000_000_000], dtype=np.int64)
    assert ak.var(ak.Array(data)) == pytest.approx(np.var(data.astype(np.float64)))
    assert not np.isnan(ak.std(ak.Array(data)))


def test_var_float32_precision():
    # float32 squares lose precision / overflow; float64 accumulation matches.
    data = np.array([1e20, 2e20, 3e20], dtype=np.float32)
    result = ak.var(ak.Array(data))
    assert result == pytest.approx(np.var(data.astype(np.float64)), rel=1e-6)


def test_var_jagged_and_axis_none_agree():
    array = ak.values_astype(
        ak.Array([[100000, 200000, 300000], [], [400000, 500000]]), np.int32
    )
    per_list = ak.var(array, axis=-1)
    assert per_list[0] == pytest.approx(np.var([100000.0, 200000.0, 300000.0]))
    assert np.isnan(per_list[1])
    assert per_list[2] == pytest.approx(np.var([400000.0, 500000.0]))

    flat = ak.values_astype(ak.Array([100000, 200000, 300000]), np.int32)
    assert ak.var(flat) == pytest.approx(np.var([100000.0, 200000.0, 300000.0]))


def test_var_bool():
    array = ak.Array([[True, False, True], [True, True]])
    result = ak.var(array, axis=-1)
    assert result[0] == pytest.approx(np.var([1.0, 0.0, 1.0]))
    assert result[1] == pytest.approx(0.0)


def test_var_typetracer_is_float64():
    base = ak.values_astype(ak.Array([[1, 2, 3], [4, 5]]), np.int32)
    tt = ak.to_backend(base, "typetracer")
    assert str(ak.var(tt, axis=-1).type) == "2 * float64"


def test_var_float64_unchanged():
    data = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    assert ak.var(ak.Array(data)) == pytest.approx(np.var(data))


def _np_weighted_var(x, w):
    x = x.astype(np.float64)
    w = w.astype(np.float64)
    avg = np.average(x, weights=w)
    return np.average((x - avg) ** 2, weights=w)


def test_weighted_var_int32_overflow():
    # x*x*weight overflows int32; promoting x to float64 keeps the products safe.
    x = np.array([100000, 200000, 300000], dtype=np.int32)
    w = np.array([1, 2, 3], dtype=np.int32)
    result = ak.var(ak.Array(x), weight=ak.Array(w))
    assert result == pytest.approx(_np_weighted_var(x, w))


def test_weighted_var_float32_precision():
    x = np.array([1e20, 2e20, 3e20], dtype=np.float32)
    w = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = ak.var(ak.Array(x), weight=ak.Array(w))
    assert result == pytest.approx(_np_weighted_var(x, w), rel=1e-5)


def test_weighted_var_jagged():
    x = ak.values_astype(
        ak.Array([[100000, 200000, 300000], [400000, 500000]]), np.int32
    )
    w = ak.values_astype(ak.Array([[1, 2, 3], [1, 1]]), np.int32)
    result = ak.var(x, weight=w, axis=-1)
    assert result[0] == pytest.approx(
        _np_weighted_var(np.array([100000, 200000, 300000]), np.array([1, 2, 3]))
    )
    assert result[1] == pytest.approx(
        _np_weighted_var(np.array([400000, 500000]), np.array([1, 1]))
    )


def test_weighted_std_int32_overflow():
    x = np.array([100000, 200000, 300000], dtype=np.int32)
    w = np.array([1, 2, 3], dtype=np.int32)
    result = ak.std(ak.Array(x), weight=ak.Array(w))
    assert result == pytest.approx(np.sqrt(_np_weighted_var(x, w)))


def test_var_complex_preserved():
    # Complex is not covered by the float64 sum-of-squares reducer; var keeps the
    # original sum(x*x) path, yielding a complex E[x**2] - E[x]**2 (unchanged).
    data = np.array([1 + 2j, 3 + 4j, 5 + 1j])
    result = ak.var(ak.Array(data))
    expected = np.mean(data**2) - np.mean(data) ** 2
    assert complex(result) == pytest.approx(complex(expected))


def test_var_uint_no_overflow():
    data = np.array([100000, 200000, 300000], dtype=np.uint32)
    assert ak.var(ak.Array(data)) == pytest.approx(np.var(data.astype(np.float64)))


def test_var_negative_values():
    data = np.array([-100000, 200000, -300000], dtype=np.int32)
    assert ak.var(ak.Array(data)) == pytest.approx(np.var(data.astype(np.float64)))


def test_std_int64_overflow():
    data = np.array([3_000_000_000, 4_000_000_000, 5_000_000_000], dtype=np.int64)
    result = ak.std(ak.Array(data))
    assert result == pytest.approx(np.std(data.astype(np.float64)))
    assert not np.isnan(result)


def test_var_keepdims():
    array = ak.values_astype(ak.Array([[100000, 200000, 300000]]), np.int32)
    out = ak.var(array, axis=-1, keepdims=True)
    assert out.to_list() == [[pytest.approx(np.var([1e5, 2e5, 3e5]))]]


def test_var_mask_identity_empty():
    array = ak.values_astype(ak.Array([[100000, 200000], [], [300000]]), np.int32)
    out = ak.var(array, axis=-1, mask_identity=True)
    assert out[1] is None
    assert out[0] == pytest.approx(np.var([1e5, 2e5]))


def test_nanvar_integer():
    data = np.array([100000, 200000, 300000], dtype=np.int32)
    assert ak.nanvar(ak.Array(data)) == pytest.approx(np.var(data.astype(np.float64)))
    assert ak.nanstd(ak.Array(data)) == pytest.approx(np.std(data.astype(np.float64)))


def test_sumofsquares_reducer_rejects_complex():
    # The float64 reducer cannot represent complex; it raises before dispatch
    # (var routes complex through the legacy sum(x*x) path instead).
    layout = ak.Array(np.array([1 + 2j, 3 + 4j])).layout
    with pytest.raises(TypeError):
        ak._do.reduce(
            layout,
            ak._reducers.SumOfSquares(),
            axis=None,
            mask=False,
            keepdims=False,
        )
