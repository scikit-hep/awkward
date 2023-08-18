# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak
from awkward._nplikes.shape import unknown_length


def test():
    fromjson = {
        "class": "RecordArray",
        "fields": ["muon", "anindex"],
        "contents": [
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
                        {
                            "class": "NumpyArray",
                            "primitive": "int64",
                            "inner_shape": [],
                            "parameters": {},
                            "form_key": "muon_thing1!",
                        },
                    ],
                    "parameters": {},
                    "form_key": "muon_record!",
                },
                "parameters": {},
                "form_key": "muon_list_outer!",
            },
            {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": "anindex!",
            },
        ],
        "parameters": {},
        "form_key": "outer!",
    }

    form = ak.forms.from_dict(fromjson)

    ttlayout = form.length_zero_array(highlevel=False).to_typetracer(forget_length=True)

    ttarray = ak.Array(ttlayout, behavior=ak.behavior)
    backend = ttlayout.backend

    result = ak.unflatten(ttarray.muon.pt, ttarray.anindex, axis=1)
    assert result.layout.is_equal_to(
        ak.contents.ListOffsetArray(
            ak.index.Index64(
                backend.index_nplike.empty(unknown_length, dtype=np.int64)
            ),
            ak.contents.ListOffsetArray(
                ak.index.Index64(
                    backend.index_nplike.empty(unknown_length, dtype=np.int64)
                ),
                ak.contents.NumpyArray(
                    backend.nplike.empty(unknown_length, dtype=np.int64)
                ),
            ),
        )
    )
