# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import json

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak
from awkward._nplikes.shape import unknown_length

form_dict = {
    "class": "RecordArray",
    "fields": ["x", "y"],
    "contents": [
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": "x.list.content",
            },
            "parameters": {},
            "form_key": "x.list.offsets",
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": "y.list.content",
            },
            "parameters": {},
            "form_key": "y.list.offsets",
        },
    ],
    "parameters": {},
}


def test_layout():
    form = ak.forms.from_dict(form_dict)
    layout, report = ak.typetracer.typetracer_with_report(form)
    assert isinstance(layout, ak.contents.Content)
    array = ak.Array(layout)

    y = array.y
    assert len(report.data_touched) == 0
    assert len(report.shape_touched) == 0

    ak.sum(y)
    assert set(report.data_touched) == {"y.list.offsets", "y.list.content"}
    assert set(report.shape_touched) == {"y.list.offsets", "y.list.content"}


def test_layout_highlevel_false():
    form = ak.forms.from_dict(form_dict)
    layout, report = ak.typetracer.typetracer_with_report(form, highlevel=False)
    assert isinstance(layout, ak.contents.Content)
    array = ak.Array(layout)

    y = array.y
    assert len(report.data_touched) == 0
    assert len(report.shape_touched) == 0

    ak.sum(y)
    assert set(report.data_touched) == {"y.list.offsets", "y.list.content"}
    assert set(report.shape_touched) == {"y.list.offsets", "y.list.content"}


def test_array_highlevel_true():
    form = ak.forms.from_dict(form_dict)
    array, report = ak.typetracer.typetracer_with_report(form, highlevel=True)
    assert isinstance(array, ak.Array)

    y = array.y
    assert len(report.data_touched) == 0
    assert len(report.shape_touched) == 0

    ak.sum(y)
    assert set(report.data_touched) == {"y.list.offsets", "y.list.content"}
    assert set(report.shape_touched) == {"y.list.offsets", "y.list.content"}


def test_form_dict():
    layout, report = ak.typetracer.typetracer_with_report(form_dict)
    assert isinstance(layout, ak.contents.Content)
    array = ak.Array(layout)

    y = array.y
    assert len(report.data_touched) == 0
    assert len(report.shape_touched) == 0

    ak.sum(y)
    assert set(report.data_touched) == {"y.list.offsets", "y.list.content"}
    assert set(report.shape_touched) == {"y.list.offsets", "y.list.content"}


def test_form_str():
    layout, report = ak.typetracer.typetracer_with_report(json.dumps(form_dict))
    assert isinstance(layout, ak.contents.Content)
    array = ak.Array(layout)

    y = array.y
    assert len(report.data_touched) == 0
    assert len(report.shape_touched) == 0

    ak.sum(y)
    assert set(report.data_touched) == {"y.list.offsets", "y.list.content"}
    assert set(report.shape_touched) == {"y.list.offsets", "y.list.content"}


def test_typetracer_from_form_highlevel_false():
    layout = ak.typetracer.typetracer_from_form(form_dict, highlevel=False)
    assert isinstance(layout, ak.contents.Content)
    assert layout.backend.name == "typetracer"
    assert layout.length is unknown_length


def test_typetracer_from_form_highlevel_true():
    array = ak.typetracer.typetracer_from_form(form_dict, highlevel=True)
    assert isinstance(array, ak.Array)
    assert ak.backend(array) == "typetracer"
    assert array.layout.length is unknown_length
