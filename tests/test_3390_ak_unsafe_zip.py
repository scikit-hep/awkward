# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
# ruff: noqa: E402

from __future__ import annotations

import awkward as ak


def test_ak_unsafe_zip_NumpyArray_dict():
    a = ak.Array([1])
    b = ak.Array([2])
    c = ak.unsafe_zip({"a": a, "b": b})
    assert ak.to_list(c) == ak.to_list(ak.zip({"a": a, "b": b}))


def test_ak_unsafe_zip_ListOffsetArray_dict():
    a = ak.Array([[1], []])
    b = ak.Array([[2], []])
    c = ak.unsafe_zip({"a": a, "b": b})
    assert ak.to_list(c) == ak.to_list(ak.zip({"a": a, "b": b}))


def test_ak_unsafe_zip_NumpyArray_list():
    a = ak.Array([1])
    b = ak.Array([2])
    c = ak.unsafe_zip([a, b])
    assert ak.to_list(c) == ak.to_list(ak.zip([a, b]))


def test_ak_unsafe_zip_ListOffsetArray_list():
    a = ak.Array([[1], []])
    b = ak.Array([[2], []])
    c = ak.unsafe_zip([a, b])
    assert ak.to_list(c) == ak.to_list(ak.zip([a, b]))


def test_typetracer_NumpyArray_non_touching():
    tracer = ak.Array([1], backend="typetracer")

    tracer, report = ak.typetracer.typetracer_with_report(
        tracer.layout.form_with_key(), highlevel=True
    )

    _ = ak.unsafe_zip({"foo": tracer, "bar": tracer})
    assert len(report.shape_touched) == 1
    assert len(report.data_touched) == 0


def test_typetracer_ListOffsetArray_non_touching():
    tracer = ak.Array([[1], [], [2, 3]], backend="typetracer")

    tracer, report = ak.typetracer.typetracer_with_report(
        tracer.layout.form_with_key(), highlevel=True
    )

    _ = ak.unsafe_zip({"foo": tracer, "bar": tracer})
    assert len(report.shape_touched) == 1
    assert len(report.data_touched) == 0
