# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def delta_r2(a, b):
    return (a.eta - b.eta) ** 2 + (a.phi - b.phi) ** 2


def test():
    form = ak.forms.from_dict(
        {
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
                        "fields": ["pt", "eta", "phi", "crossref", "thing1"],
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
    )

    ttlayout, report = ak.typetracer.typetracer_with_report(form)

    ttarray = ak.Array(ttlayout)

    a = ak.cartesian([ttarray.muon, ttarray.jet], axis=1, nested=True)

    mval = delta_r2(a["0"], a["1"])

    assert set(report.data_touched) == {
        "muon_list!",
        "jet_list!",
        "muon_eta!",
        "jet_eta!",
        "muon_phi!",
        "jet_phi!",
    }

    mmin = ak.argmin(mval, axis=2)
    assert set(report.data_touched) == {
        "muon_list!",
        "jet_list!",
        "muon_eta!",
        "jet_eta!",
        "muon_phi!",
        "jet_phi!",
    }

    ak.firsts(a["1"][mmin], axis=2)

    assert set(report.data_touched) == {
        "muon_list!",
        "jet_list!",
        "muon_eta!",
        "jet_eta!",
        "muon_phi!",
        "jet_phi!",
    }
