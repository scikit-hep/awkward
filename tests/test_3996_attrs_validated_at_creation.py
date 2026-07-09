# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak
from awkward._attrs import Attrs


def test_attrs_wrapped_in_Attrs_at_creation():
    attrs = {"a": 1, "b": 2}

    array = ak.Array([1, 2, 3], attrs=attrs)
    assert isinstance(array.attrs, Attrs)
    assert isinstance(array._attrs, Attrs)
    assert dict(array.attrs) == attrs

    record = ak.Record({"x": 1}, attrs=attrs)
    assert isinstance(record.attrs, Attrs)
    assert isinstance(record._attrs, Attrs)
    assert dict(record.attrs) == attrs

    builder = ak.ArrayBuilder(attrs=attrs)
    assert isinstance(builder.attrs, Attrs)
    assert isinstance(builder._attrs, Attrs)
    assert dict(builder.attrs) == attrs


def test_attrs_isolated_from_original_dict():
    attrs = {"a": 1}
    array = ak.Array([1, 2, 3], attrs=attrs)
    record = ak.Record({"x": 1}, attrs=attrs)
    builder = ak.ArrayBuilder(attrs=attrs)

    attrs["a"] = 999

    assert array.attrs["a"] == 1
    assert record.attrs["a"] == 1
    assert builder.attrs["a"] == 1


def test_non_string_keys_rejected_at_creation():
    bad_attrs = {1: 2}

    with pytest.raises(TypeError, match="'attrs' keys must be strings"):
        ak.Array([1, 2, 3], attrs=bad_attrs)
    with pytest.raises(TypeError, match="'attrs' keys must be strings"):
        ak.Record({"x": 1}, attrs=bad_attrs)
    with pytest.raises(TypeError, match="'attrs' keys must be strings"):
        ak.ArrayBuilder(attrs=bad_attrs)


def test_non_mapping_attrs_rejected_at_creation():
    with pytest.raises(TypeError, match="attrs must be None or a mapping"):
        ak.Array([1, 2, 3], attrs=[("a", 1)])
    with pytest.raises(TypeError, match="attrs must be None or a mapping"):
        ak.Record({"x": 1}, attrs=[("a", 1)])
    with pytest.raises(TypeError, match="attrs must be None or a mapping"):
        ak.ArrayBuilder(attrs=[("a", 1)])


def test_attrs_setter_roundtrip():
    array = ak.Array([1, 2, 3], attrs={"a": 1})
    array.attrs = array.attrs
    assert isinstance(array.attrs, Attrs)
    assert dict(array.attrs) == {"a": 1}

    record = ak.Record({"x": 1}, attrs={"a": 1})
    record.attrs = record.attrs
    assert isinstance(record.attrs, Attrs)
    assert dict(record.attrs) == {"a": 1}

    builder = ak.ArrayBuilder(attrs={"a": 1})
    builder.attrs = builder.attrs
    assert isinstance(builder.attrs, Attrs)
    assert dict(builder.attrs) == {"a": 1}
