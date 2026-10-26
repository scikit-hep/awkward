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


def test_caches_default_without_init():
    array = ak.Array.__new__(ak.Array)
    assert array._numbaview is None
    assert array._cpp_type is None
    assert array._generator is None
    assert array._lookup is None
    assert ak.Record.__new__(ak.Record)._numbaview is None


def _make(wrapper, cls):
    layout = ak.with_name([{"x": 1}], "custom", highlevel=False)
    if wrapper is ak.Array:
        return cls(layout, behavior={("*", "custom"): cls})
    return cls(layout[0], behavior={"custom": cls})


@pytest.mark.parametrize("wrapper", [ak.Array, ak.Record])
def test_setattr_override_sees_init(wrapper):
    class Logged(wrapper):
        def __setattr__(self, name, value):
            self.__dict__.setdefault("_seen", []).append(name)
            super().__setattr__(name, value)

    thing = _make(wrapper, Logged)
    assert type(thing) is Logged
    assert {"_layout", "_behavior", "_attrs", "_numbaview"} <= set(thing._seen)


@pytest.mark.parametrize("wrapper", [ak.Array, ak.Record])
def test_setattr_of_later_mixin_is_called(wrapper):
    class Mixin:
        def __setattr__(self, name, value):
            self.__dict__.setdefault("_seen", []).append(name)
            object.__setattr__(self, name, value)

    class Mixed(wrapper, Mixin): ...

    thing = _make(wrapper, Mixed)
    assert type(thing) is Mixed
    assert "_layout" in thing._seen
    thing._foo = 1
    assert thing._seen[-1] == "_foo"
    assert thing._foo == 1


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


def test_record_awkward1_setstate(point_behavior):
    # awkward 1 pickled a record as (form, length, container, behavior, at)
    array = ak.Array([{"x": 1, "y": 2}, {"x": 3, "y": 4}], with_name="point")
    form, length, container = ak.to_buffers(
        array, buffer_key="{form_key}-{attribute}", form_key="node{id}"
    )

    record = ak.Record.__new__(ak.Record)
    record.__setstate__((form.to_dict(), length, container, None, 1))
    assert isinstance(record, PointRecord)
    assert record.to_list() == {"x": 3, "y": 4}
    assert record._attrs is None
    assert record._numbaview is None


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
