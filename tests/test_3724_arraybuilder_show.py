from __future__ import annotations

import io

import awkward as ak


def _capture_show_output(obj):
    stream = io.StringIO()
    obj.show(stream=stream)
    return stream.getvalue()


def test_arraybuilder_show_simple_list():
    builder = ak.ArrayBuilder()
    builder.begin_list()
    builder.real(1.2)
    builder.real(3.4)
    builder.end_list()

    out_builder = _capture_show_output(builder)
    out_snapshot = _capture_show_output(builder.snapshot())

    assert out_builder == out_snapshot


def test_arraybuilder_show_record():
    builder = ak.ArrayBuilder()
    builder.begin_record()
    builder.field("x").real(1.2)
    builder.field("y").integer(3)
    builder.end_record()

    out_builder = _capture_show_output(builder)
    out_snapshot = _capture_show_output(builder.snapshot())

    assert out_builder == out_snapshot


def test_arraybuilder_show_nested_list():
    builder = ak.ArrayBuilder()
    builder.begin_list()
    builder.begin_list()
    builder.integer(1)
    builder.integer(2)
    builder.end_list()
    builder.end_list()

    out_builder = _capture_show_output(builder)
    out_snapshot = _capture_show_output(builder.snapshot())

    assert out_builder == out_snapshot
