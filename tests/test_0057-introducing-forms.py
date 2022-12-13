# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json
import pickle

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


def test_forms():
    form = ak.forms.NumpyForm("float64")
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak.forms.from_json(form.to_json()) == form
    assert form.inner_shape == ()
    assert form.itemsize == 8
    assert form.primitive == "float64"
    assert form.parameters == {}
    assert form.form_key is None
    assert json.loads(form.to_json()) == {
        "class": "NumpyArray",
        "inner_shape": [],
        "primitive": "float64",
        "parameters": {},
        "form_key": None,
    }
    assert json.loads(str(form)) == {
        "class": "NumpyArray",
        "primitive": "float64",
    }

    form = ak.forms.NumpyForm(
        "float64",
        [1, 2, 3],
        parameters={"hey": ["you", {"there": 3}]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak.forms.from_json(form.to_json()) == form
    assert form.inner_shape == (1, 2, 3)
    assert form.itemsize == 8
    assert form.primitive == "float64"
    assert form.parameters == {"hey": ["you", {"there": 3}]}
    assert form.form_key == "yowzers"
    assert json.loads(form.to_json()) == {
        "class": "NumpyArray",
        "inner_shape": [1, 2, 3],
        "primitive": "float64",
        "parameters": {"hey": ["you", {"there": 3}]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "NumpyArray",
        "inner_shape": [1, 2, 3],
        "primitive": "float64",
        "parameters": {"hey": ["you", {"there": 3}]},
        "form_key": "yowzers",
    }

    form = ak.forms.BitMaskedForm(
        "i8",
        ak.forms.NumpyForm("float64"),
        True,
        False,
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "BitMaskedArray",
        "mask": "i8",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "parameters": {},
            "form_key": None,
        },
        "valid_when": True,
        "lsb_order": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "BitMaskedArray",
        "mask": "i8",
        "content": "float64",
        "valid_when": True,
        "lsb_order": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak.forms.EmptyForm(
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "EmptyArray",
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "EmptyArray",
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak.forms.IndexedForm(
        "i64",
        ak.forms.NumpyForm("float64"),
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "IndexedArray",
        "index": "i64",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "IndexedArray",
        "index": "i64",
        "content": "float64",
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak.forms.IndexedOptionForm(
        "i64",
        ak.forms.NumpyForm("float64"),
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "IndexedOptionArray",
        "index": "i64",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "IndexedOptionArray",
        "index": "i64",
        "content": "float64",
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak.forms.ListForm(
        "i64",
        "i64",
        ak.forms.NumpyForm("float64"),
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "ListArray",
        "starts": "i64",
        "stops": "i64",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "ListArray",
        "starts": "i64",
        "stops": "i64",
        "content": "float64",
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak.forms.ListOffsetForm(
        "i64",
        ak.forms.NumpyForm("float64"),
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "ListOffsetArray",
        "offsets": "i64",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "ListOffsetArray",
        "offsets": "i64",
        "content": "float64",
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak.forms.RecordForm(
        [ak.forms.NumpyForm("float64"), ak.forms.NumpyForm("bool")],
        ["one", "two"],
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "RecordArray",
        "fields": ["one", "two"],
        "contents": [
            {
                "class": "NumpyArray",
                "inner_shape": [],
                "primitive": "float64",
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "inner_shape": [],
                "primitive": "bool",
                "parameters": {},
                "form_key": None,
            },
        ],
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "RecordArray",
        "fields": ["one", "two"],
        "contents": [
            "float64",
            "bool",
        ],
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak.forms.RecordForm(
        [ak.forms.NumpyForm("float64"), ak.forms.NumpyForm("bool")],
        None,
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "RecordArray",
        "fields": None,
        "contents": [
            {
                "class": "NumpyArray",
                "inner_shape": [],
                "primitive": "float64",
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "inner_shape": [],
                "primitive": "bool",
                "parameters": {},
                "form_key": None,
            },
        ],
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "RecordArray",
        "fields": None,
        "contents": ["float64", "bool"],
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak.forms.RegularForm(
        ak.forms.NumpyForm("float64"),
        10,
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "RegularArray",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "parameters": {},
            "form_key": None,
        },
        "size": 10,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "RegularArray",
        "content": "float64",
        "size": 10,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak.forms.UnionForm(
        "i8",
        "i64",
        [ak.forms.NumpyForm("float64"), ak.forms.NumpyForm("bool")],
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i64",
        "contents": [
            {
                "class": "NumpyArray",
                "inner_shape": [],
                "primitive": "float64",
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "inner_shape": [],
                "primitive": "bool",
                "parameters": {},
                "form_key": None,
            },
        ],
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i64",
        "contents": ["float64", "bool"],
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak.forms.UnmaskedForm(
        ak.forms.NumpyForm("float64"),
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "UnmaskedArray",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "parameters": {},
            "form_key": None,
        },
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "UnmaskedArray",
        "content": "float64",
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
