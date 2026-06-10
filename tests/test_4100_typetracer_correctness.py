from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.shape import unknown_length
from awkward._nplikes.typetracer import TypeTracer, TypeTracerArray


@pytest.fixture
def tt():
    return TypeTracer.instance()


def test_min_max_sum_axis_none_terminate(tt):
    x = TypeTracerArray._new(np.dtype(np.int32), (unknown_length,))
    assert tt.min(x, axis=None).dtype == np.dtype(np.int32)
    assert tt.max(x, axis=None).dtype == np.dtype(np.int32)
    assert tt.min(x, axis=None).shape == ()
    # sum promotes int32 -> platform int
    out = tt.sum(x, axis=None)
    assert out.shape == ()
    assert out.dtype == np.empty(0, dtype=np.int32).sum().dtype


def test_sum_dtype_promotion_bool(tt):
    x = TypeTracerArray._new(np.dtype(np.bool_), (unknown_length,))
    out = tt.sum(x, axis=None)
    assert out.dtype == np.empty(0, dtype=np.bool_).sum().dtype
    assert out.dtype != np.dtype(np.bool_)


def test_sum_keepdims(tt):
    x = TypeTracerArray._new(np.dtype(np.int64), (unknown_length, 3))
    out = tt.sum(x, axis=1, keepdims=True)
    assert out.shape == (unknown_length, 1)
    out2 = tt.sum(x, axis=1)
    assert out2.shape == (unknown_length,)


def test_stack_basic(tt):
    a = TypeTracerArray._new(np.dtype(np.float64), (unknown_length,))
    b = TypeTracerArray._new(np.dtype(np.float64), (unknown_length,))
    out = tt.stack([a, b], axis=0)
    assert out.shape == (2, unknown_length)
    assert out.dtype == np.dtype(np.float64)


def test_stack_axis_negative(tt):
    a = TypeTracerArray._new(np.dtype(np.float64), (unknown_length,))
    b = TypeTracerArray._new(np.dtype(np.float64), (unknown_length,))
    out = tt.stack([a, b], axis=-1)
    assert out.shape == (unknown_length, 2)


def test_stack_dtype_promotion(tt):
    a = TypeTracerArray._new(np.dtype(np.int32), (5,))
    b = TypeTracerArray._new(np.dtype(np.float64), (5,))
    out = tt.stack([a, b], axis=0)
    assert out.shape == (2, 5)
    assert out.dtype == np.result_type(np.int32, np.float64)


def test_searchsorted_dtype(tt):
    x = TypeTracerArray._new(np.dtype(np.float64), (unknown_length,))
    values = TypeTracerArray._new(np.dtype(np.float64), (unknown_length,))
    out = tt.searchsorted(x, values)
    assert out.dtype == np.dtype(np.intp)


def test_asarray_honors_dtype(tt):
    raw = np.array([1, 2, 3], dtype=np.int32)
    out = tt.asarray(raw, dtype=np.float64)
    assert out.dtype == np.dtype(np.float64)


def test_asarray_no_dtype_keeps_obj_dtype(tt):
    raw = np.array([1, 2, 3], dtype=np.int16)
    out = tt.asarray(raw, copy=False)
    assert out.dtype == np.dtype(np.int16)


def test_asarray_copy_false_mismatch_raises(tt):
    raw = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(ValueError):
        tt.asarray(raw, dtype=np.float64, copy=False)


def test_reshape_with_zero(tt):
    x = TypeTracerArray._new(np.dtype(np.float64), (0,))
    out = tt.reshape(x, (0, 3))
    assert out.shape == (0, 3)


def test_repr_does_not_touch_report(tt):
    from awkward._nplikes.typetracer import TypeTracerReport

    report = TypeTracerReport()
    report.set_labels(["a"])
    x = TypeTracerArray._new(
        np.dtype(np.float64), (unknown_length,), form_key="a", report=report
    )
    before = report.shape_touched
    repr(x)
    str(x)
    assert report.shape_touched == before


def test_pad_none_typetracer():
    arr = ak.Array([1, 2, 3, 4, 5])
    tt_layout = arr.layout.to_typetracer(forget_length=True)
    out = ak.pad_none(ak.Array(tt_layout), 3, axis=0)
    assert out is not None


def test_combinations_axis0_typetracer():
    arr = ak.Array([1, 2, 3, 4, 5])
    tt_layout = arr.layout.to_typetracer(forget_length=True)
    out = ak.combinations(ak.Array(tt_layout), 2, axis=0)
    assert out is not None
