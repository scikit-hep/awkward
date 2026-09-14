# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import copy
import pickle

import pytest

import awkward as ak
from awkward._attrs import Attrs


class PointArray(ak.Array): ...


class PointRecord(ak.Record): ...


@pytest.fixture
def point_behavior():
    ak.behavior["point"] = PointRecord
    ak.behavior["*", "point"] = PointArray
    yield
    del ak.behavior["point"]
    del ak.behavior["*", "point"]


def test_array_init_state():
    layout = ak.to_layout([[1, 2, 3], [], [4, 5]])

    from_layout = ak.Array(layout)
    assert from_layout.layout is layout
    assert from_layout.behavior is None
    assert from_layout._attrs is None
    assert from_layout.attrs == {}

    from_iterable = ak.Array([[1, 2, 3], [], [4, 5]])
    assert from_iterable.to_list() == [[1, 2, 3], [], [4, 5]]
    assert from_iterable.behavior is None
    assert from_iterable._attrs is None

    behavior, attrs = {"kind": "test"}, {"meta": 123}
    explicit = ak.Array(layout, behavior=behavior, attrs=attrs)
    assert explicit.behavior is behavior
    assert isinstance(explicit.attrs, Attrs)
    assert dict(explicit.attrs) == attrs

    from_array = ak.Array(explicit)
    assert from_array.layout is layout
    assert from_array.behavior == behavior
    assert dict(from_array.attrs) == attrs


def test_record_init_state():
    layout = ak.Array([{"x": 1, "y": 2}]).layout[0]

    from_layout = ak.Record(layout)
    assert from_layout.layout is layout
    assert from_layout.behavior is None
    assert from_layout._attrs is None
    assert from_layout.attrs == {}

    from_dict = ak.Record({"x": 1, "y": 2})
    assert from_dict.to_list() == {"x": 1, "y": 2}
    assert from_dict.behavior is None
    assert from_dict._attrs is None

    behavior, attrs = {"kind": "test"}, {"meta": 123}
    explicit = ak.Record(layout, behavior=behavior, attrs=attrs)
    assert explicit.behavior is behavior
    assert isinstance(explicit.attrs, Attrs)
    assert dict(explicit.attrs) == attrs

    from_record = ak.Record(explicit)
    assert from_record.layout is layout
    assert from_record.behavior == behavior
    assert dict(from_record.attrs) == attrs


@pytest.mark.parametrize("cls", [ak.Array, ak.Record])
def test_numbaview_class_default(cls):
    # the default lives on the class, so instances need not define it
    assert vars(cls)["_numbaview"] is None
    uninitialised = cls.__new__(cls)
    assert "_numbaview" not in uninitialised.__dict__
    assert uninitialised._numbaview is None


def test_numbaview_reset_on_new_layout():
    array = ak.Array([[1, 2, 3], [], [4, 5]])
    assert array._numbaview is None
    array._numbaview = "stale"
    array.layout = ak.to_layout([[1, 2, 3, 4]])
    assert array._numbaview is None
    assert array.to_list() == [[1, 2, 3, 4]]

    record = ak.Record({"x": 1})
    assert record._numbaview is None
    record._numbaview = "stale"
    record.layout = ak.Array([{"x": 2}]).layout[0]
    assert record._numbaview is None
    assert record.to_list() == {"x": 2}


@pytest.mark.parametrize(
    "roundtrip", [copy.copy, copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))]
)
def test_array_roundtrip(roundtrip, point_behavior):
    array = ak.Array([[{"x": 1, "y": 2}], []], with_name="point", attrs={"meta": 123})
    assert isinstance(array, PointArray)

    result = roundtrip(array)
    assert isinstance(result, PointArray)
    assert result.to_list() == array.to_list()
    assert dict(result.attrs) == {"meta": 123}
    assert result._numbaview is None


@pytest.mark.parametrize(
    "roundtrip", [copy.copy, copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))]
)
def test_record_roundtrip(roundtrip, point_behavior):
    record = ak.Array([{"x": 1, "y": 2}], with_name="point")[0]
    assert isinstance(record, PointRecord)

    result = roundtrip(record)
    assert isinstance(result, PointRecord)
    assert result.to_list() == {"x": 1, "y": 2}
    assert result._numbaview is None


@pytest.mark.parametrize("wrapper", [ak.Array, ak.Record])
def test_setattr_still_guarded(wrapper):
    layout = ak.to_layout([{"x": 1}])
    thing = wrapper(layout) if wrapper is ak.Array else wrapper(layout[0])

    with pytest.raises(AttributeError, match="only private attributes"):
        thing.not_a_field = 1

    with pytest.raises(AttributeError, match="fields cannot be set as attributes"):
        thing.x = 1

    thing._private = 1
    assert thing._private == 1


def test_layout_setter_still_works():
    array = ak.Array([1, 2, 3])
    array.layout = ak.to_layout([1, 2, 3, 4])
    assert array.to_list() == [1, 2, 3, 4]

    with pytest.raises(TypeError, match="layout must be a subclass"):
        array.layout = [1, 2, 3]
