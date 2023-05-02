# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak


def test():
    form = ak.forms.from_dict(
        {
            "class": "RecordArray",
            "fields": ["muon"],
            "contents": [
                {
                    "class": "ListOffsetArray",
                    "offsets": "i64",
                    "content": {
                        "class": "RecordArray",
                        "fields": ["pt"],
                        "contents": [
                            {
                                "class": "NumpyArray",
                                "primitive": "int64",
                                "inner_shape": [],
                                "parameters": {},
                                "form_key": "muon_pt!",
                            },
                        ],
                        "parameters": {},
                        "form_key": "muon_record!",
                    },
                    "parameters": {},
                    "form_key": "muon_list_outer!",
                },
            ],
            "parameters": {},
            "form_key": "outer!",
        }
    )

    ttlayout, report = ak._nplikes.typetracer.typetracer_with_report(
        form, forget_length=True
    )
    ttarray = ak.Array(ttlayout)
    array = ak.Array(ttlayout.form.length_zero_array(highlevel=False))
    array["emptydict"] = {}
    array["emptylist"] = []
    ttarray["emptydict"] = {}
    ttarray["emptylist"] = []

    assert ttarray.layout.form == array.layout.form
