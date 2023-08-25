# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


def test():
    form = ak.forms.from_dict(
        {
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
    )
    layout, report = ak.typetracer.typetracer_with_report(form)
    array = ak.Array(layout)

    y = array.y
    assert len(report.data_touched) == 0
    assert len(report.shape_touched) == 0

    ak.typetracer.touch_data(y)
    assert len(report.data_touched) == 2
    assert len(report.shape_touched) == 2
