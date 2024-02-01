# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest  # noqa: F401

import awkward as ak

form_dict = {
    "class": "RecordArray",
    "fields": ["x"],
    "contents": [
        {
            "class": "ListArray",
            "starts": "i64",
            "stops": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": "x.list.content",
            },
            "parameters": {},
            "form_key": "x.list",
        }
    ],
    "parameters": {},
}


def test_without_attribute():
    form = ak.forms.from_dict(form_dict)
    array, report = ak.typetracer.typetracer_with_report(
        form, buffer_key="{form_key}", highlevel=True
    )
    ak.typetracer.touch_data(array)
    assert set(report.data_touched) == {
        "x.list",
        "x.list.content",
    }


def test_with_attribute():
    form = ak.forms.from_dict(form_dict)
    array, report = ak.typetracer.typetracer_with_report(
        form, buffer_key="{form_key}-{attribute}", highlevel=True
    )
    ak.typetracer.touch_data(array)
    assert set(report.data_touched) == {
        "x.list-starts",
        "x.list-stops",
        "x.list.content-data",
    }
