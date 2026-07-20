from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.shape import unknown_length
from awkward._nplikes.typetracer import TypeTracer, TypeTracerArray


@pytest.fixture
def tt():
    return TypeTracer.instance()


def test_min_axis_none_terminate(tt):
    x = TypeTracerArray._new(np.dtype(np.int32), (unknown_length,))
    out = tt.min(x, axis=None)
    assert out.dtype == np.dtype(np.int32)
    assert out.shape == ()


def test_sum_dtype_promotion_bool(tt):
    x = TypeTracerArray._new(np.dtype(np.bool_), (unknown_length,))
    out = tt.sum(x, axis=None)
    assert out.shape == ()
    assert out.dtype == np.empty(0, dtype=np.bool_).sum().dtype
    assert out.dtype != np.dtype(np.bool_)


def test_stack_symbolic_shape(tt):
    a = TypeTracerArray._new(np.dtype(np.int32), (unknown_length,))
    b = TypeTracerArray._new(np.dtype(np.float64), (unknown_length,))
    out = tt.stack([a, b], axis=0)
    assert out.shape == (2, unknown_length)
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
