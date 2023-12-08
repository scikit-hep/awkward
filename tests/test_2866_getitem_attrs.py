# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest  # noqa: F401

import awkward as ak

ATTRS = {"foo": "bar", "@foo": "baz"}


def test_array_slice():
    array = ak.Array([[0, 1, 2], [4]], attrs=ATTRS)
    assert array.attrs is ATTRS

    assert array[0].attrs is ATTRS
    assert array[1:].attrs is ATTRS


def test_array_field():
    array = ak.Array([[{"x": 1}, {"x": 2}], [{"x": 10}]], attrs=ATTRS)
    assert array.attrs is ATTRS

    assert array.x.attrs is ATTRS
    assert array.x[1:].attrs is ATTRS


def test_record_field():
    array = ak.Array([{"x": [1, 2, 3]}], attrs=ATTRS)
    assert array.attrs is ATTRS

    record = array[0]
    assert record.attrs is ATTRS
    assert record.x.attrs is ATTRS
