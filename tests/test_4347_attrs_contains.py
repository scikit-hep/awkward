# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest  # noqa: F401

import awkward as ak
from awkward._attrs import Attrs
from awkward._namedaxis import NAMED_AXIS_KEY


def test_contains_agrees_with_backing_dict():
    data = {"foo": "bar", "@transient": 123}
    attrs = Attrs(data)

    for key in "foo", "@transient", "missing":
        assert (key in attrs) == (key in data)

    assert "foo" in attrs
    assert "missing" not in attrs

    # non-string keys are not valid 'attrs' keys, but looking them up must not raise
    assert (1 in attrs) is False
    assert (None in attrs) is False
    assert 1 not in attrs


def test_contains_consistent_after_setitem():
    attrs = Attrs({"foo": "bar"})
    assert "baz" not in attrs
    assert len(attrs) == 1

    attrs["baz"] = 456

    assert "baz" in attrs
    assert attrs["baz"] == 456
    assert "baz" in attrs.keys()
    assert len(attrs) == 2
    assert set(attrs) == {"foo", "baz"}


def test_named_axis_lookup_survives_getitem():
    array = ak.Array([[1, 2, 3], [4, 5]], named_axis=("x", "y"), attrs={"foo": "bar"})
    assert NAMED_AXIS_KEY in array.attrs
    assert array.named_axis == {"x": 0, "y": 1}

    sliced = array[1:]

    assert NAMED_AXIS_KEY in sliced.attrs
    assert sliced.named_axis == {"x": 0, "y": 1}
    assert sliced.attrs["foo"] == "bar"
