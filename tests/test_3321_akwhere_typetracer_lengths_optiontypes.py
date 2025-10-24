from __future__ import annotations

import awkward as ak

fromdict = {
    "class": "RecordArray",
    "fields": ["muon", "jet"],
    "contents": [
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "RecordArray",
                "fields": ["pt", "eta", "phi", "crossref"],
                "contents": [
                    {
                        "class": "NumpyArray",
                        "primitive": "int64",
                        "inner_shape": [],
                        "parameters": {},
                        "form_key": "muon_pt!",
                    },
                    {
                        "class": "NumpyArray",
                        "primitive": "int64",
                        "inner_shape": [],
                        "parameters": {},
                        "form_key": "muon_eta!",
                    },
                    {
                        "class": "NumpyArray",
                        "primitive": "int64",
                        "inner_shape": [],
                        "parameters": {},
                        "form_key": "muon_phi!",
                    },
                    {
                        "class": "ListOffsetArray",
                        "offsets": "i64",
                        "content": {
                            "class": "NumpyArray",
                            "primitive": "int64",
                            "inner_shape": [],
                            "parameters": {},
                            "form_key": "muon_crossref_content!",
                        },
                        "parameters": {},
                        "form_key": "muon_crossref_index!",
                    },
                ],
                "parameters": {},
                "form_key": "muon_record!",
            },
            "parameters": {},
            "form_key": "muon_list!",
        },
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "RecordArray",
                "fields": [
                    "pt",
                    "eta",
                    "phi",
                    "crossref",
                    "thing1",
                ],
                "contents": [
                    {
                        "class": "NumpyArray",
                        "primitive": "int64",
                        "inner_shape": [],
                        "parameters": {},
                        "form_key": "jet_pt!",
                    },
                    {
                        "class": "NumpyArray",
                        "primitive": "int64",
                        "inner_shape": [],
                        "parameters": {},
                        "form_key": "jet_eta!",
                    },
                    {
                        "class": "NumpyArray",
                        "primitive": "int64",
                        "inner_shape": [],
                        "parameters": {},
                        "form_key": "jet_phi!",
                    },
                    {
                        "class": "ListOffsetArray",
                        "offsets": "i64",
                        "content": {
                            "class": "NumpyArray",
                            "primitive": "int64",
                            "inner_shape": [],
                            "parameters": {},
                            "form_key": "jet_crossref_content!",
                        },
                        "parameters": {},
                        "form_key": "jet_crossref_index!",
                    },
                    {
                        "class": "NumpyArray",
                        "primitive": "int64",
                        "inner_shape": [],
                        "parameters": {},
                        "form_key": "jet_thing1!",
                    },
                ],
                "parameters": {},
                "form_key": "jet_record!",
            },
            "parameters": {},
            "form_key": "jet_list!",
        },
    ],
    "parameters": {},
    "form_key": "outer!",
}

form = ak.forms.from_dict(fromdict)
ttlayout, report = ak.typetracer.typetracer_with_report(form)
ttarray = ak.Array(ttlayout)


def test_where():
    ak.where(abs(ttarray.jet.eta) < 1.0, 0.000511, ttarray.jet.thing1)


def test_maybe_where():
    maybe = ak.firsts(ttarray)
    ak.where(abs(maybe.jet.eta) < 1.0, 0.000511, maybe.jet.thing1)


def test_varmaybe_where():
    varmaybe = ak.pad_none(ttarray, 3)
    ak.where(abs(varmaybe.jet.eta) < 1.0, 0.000511, varmaybe.jet.thing1)
