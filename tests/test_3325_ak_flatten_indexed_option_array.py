from __future__ import annotations

import pytest

import awkward as ak

fromdict = {
    "class": "ListOffsetArray",
    "offsets": "i64",
    "content": {
        "class": "ListOffsetArray",
        "offsets": "i64",
        "content": {
            "class": "IndexedOptionArray",
            "index": "i64",
            "content": {
                "class": "ListOffsetArray",
                "offsets": "i64",
                "content": {
                    "class": "IndexedOptionArray",
                    "index": "i64",
                    "content": {
                        "class": "NumpyArray",
                        "primitive": "float32",
                        "inner_shape": [],
                        "parameters": {},
                        "form_key": None,
                    },
                    "parameters": {},
                    "form_key": None,
                },
                "parameters": {},
                "form_key": None,
            },
            "parameters": {},
            "form_key": None,
        },
        "parameters": {},
        "form_key": None,
    },
    "parameters": {},
    "form_key": None,
}

form = ak.forms.from_dict(fromdict)

ttlayout, report = ak.typetracer.typetracer_with_report(form)

ttarray = ak.Array(ttlayout)


@pytest.mark.parametrize("ax", [None, 0, 1, 2, 3])
def test_3325_flatten_index_option_array(ax):
    ak.flatten(ttarray, axis=ax)
