# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak


def test():
    # Create a dict that has .keys() view
    field_to_content = {
        "x": {
            "class": "NumpyArray",
            "primitive": "int64",
            "inner_shape": [],
            "parameters": {},
            "form_key": None,
        }
    }
    form = ak.forms.from_dict(
        {
            "class": "RecordArray",
            "fields": field_to_content.keys(),
            "contents": field_to_content.values(),
            "parameters": {},
            "form_key": None,
        }
    )
    assert isinstance(form.fields, list)
    assert form.fields == ["x"]
