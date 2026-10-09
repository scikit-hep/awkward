# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak
from awkward._layout import wrap_layout


class PointRecord(ak.Record):
    pass


class PointArray(ak.Array):
    pass


@pytest.fixture
def registered_point_behavior():
    ak.behavior["point"] = PointRecord
    ak.behavior["*", "point"] = PointArray
    try:
        yield
    finally:
        del ak.behavior["point"]
        del ak.behavior["*", "point"]


@pytest.fixture
def layout():
    return ak.Array(
        [[{"x": 1.1, "y": 1}, {"x": 2.2, "y": 2}], [], [{"x": 3.3, "y": 3}]]
    ).layout


def test_content_wraps_to_array(layout):
    wrapped = wrap_layout(layout)
    assert isinstance(wrapped, ak.highlevel.Array)
    assert wrapped.layout is layout
    assert wrapped.to_list() == layout.to_list()


def test_record_wraps_to_highlevel_record(layout):
    record = layout[0][0]
    assert isinstance(record, ak.record.Record)

    wrapped = wrap_layout(record)
    assert isinstance(wrapped, ak.highlevel.Record)
    assert wrapped.layout is record
    assert wrapped.to_list() == {"x": 1.1, "y": 1}


def test_array_builder_layout_is_passed_through():
    builder = ak.ArrayBuilder()
    builder.append(1)

    assert wrap_layout(builder._layout, allow_other=True) is builder._layout
    with pytest.raises(AssertionError):
        wrap_layout(builder._layout)


def test_highlevel_false_returns_layout_untouched(layout):
    record = layout[0][0]
    assert wrap_layout(layout, highlevel=False) is layout
    assert wrap_layout(record, highlevel=False) is record


def test_non_layout_passes_through():
    other = "not-a-layout"
    assert wrap_layout(other, allow_other=True) is other
    assert wrap_layout(other, highlevel=False, allow_other=True) is other
    with pytest.raises(AssertionError):
        wrap_layout(other)


def test_behavior_and_attrs_are_threaded(layout):
    behavior = {"some-key": "some-value"}
    attrs = {"some-attr": 123}

    wrapped = wrap_layout(layout, behavior=behavior, attrs=attrs)
    assert wrapped.behavior == behavior
    assert wrapped.attrs == attrs

    wrapped_record = wrap_layout(layout[0][0], behavior=behavior, attrs=attrs)
    assert wrapped_record.behavior == behavior
    assert wrapped_record.attrs == attrs


def test_behavior_is_taken_from_like(layout):
    like = ak.Array(layout, behavior={"some-key": "some-value"})

    assert wrap_layout(layout, like=like).behavior == {"some-key": "some-value"}
    # an explicit behavior wins over ``like``
    assert wrap_layout(layout, behavior={}, like=like).behavior == {}


def test_global_behavior_selects_subclass(layout, registered_point_behavior):
    named = ak.with_name(wrap_layout(layout), "point").layout

    assert isinstance(wrap_layout(named), PointArray)
    assert isinstance(wrap_layout(named[0][0]), PointRecord)


def test_passed_behavior_selects_subclass(layout):
    named = ak.with_name(wrap_layout(layout), "point").layout
    behavior = {"point": PointRecord, ("*", "point"): PointArray}

    assert isinstance(wrap_layout(named, behavior=behavior), PointArray)
    assert isinstance(wrap_layout(named[0][0], behavior=behavior), PointRecord)


def test_wrapped_result_drives_ordinary_operations(layout):
    array = wrap_layout(layout, attrs={"some-attr": 123})

    assert isinstance(array[0], ak.highlevel.Array)
    assert isinstance(array[0][0], ak.highlevel.Record)
    assert isinstance(array["x"], ak.highlevel.Array)
    assert isinstance(array[1:], ak.highlevel.Array)
    assert isinstance(ak.num(array), ak.highlevel.Array)

    assert ak.num(array).to_list() == [2, 0, 1]
    assert array["y"].to_list() == [[1, 2], [], [3]]
    # attrs survive the round-trip through operations
    assert ak.num(array).attrs == {"some-attr": 123}
