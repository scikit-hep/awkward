# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json

try:
    # pybind11 only supports cPickle protocol 2+ (-1 in pickle.dumps)
    # (automatically satisfied in Python 3; this is just to keep testing Python 2.7)
    import cPickle as pickle
except ImportError:
    import pickle

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2  # noqa: F401

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_forms():
    form = ak._v2.forms.NumpyForm("float64")
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert form.inner_shape == ()
    assert form.itemsize == 8
    assert form.primitive == "float64"
    assert form.has_identifier is False
    assert form.parameters == {}
    assert form.form_key is None
    assert json.loads(form.to_json()) == {
        "class": "NumpyArray",
        "inner_shape": [],
        "primitive": "float64",
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert json.loads(str(form)) == {
        "class": "NumpyArray",
        "primitive": "float64",
    }

    form = ak._v2.forms.NumpyForm(
        "float64",
        [1, 2, 3],
        has_identifier=True,
        parameters={"hey": ["you", {"there": 3}]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert form.inner_shape == (1, 2, 3)
    assert form.itemsize == 8
    assert form.primitive == "float64"
    assert form.has_identifier is True
    assert form.parameters == {"hey": ["you", {"there": 3}]}
    assert form.form_key == "yowzers"
    assert json.loads(form.to_json()) == {
        "class": "NumpyArray",
        "inner_shape": [1, 2, 3],
        "primitive": "float64",
        "has_identifier": True,
        "parameters": {"hey": ["you", {"there": 3}]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "NumpyArray",
        "inner_shape": [1, 2, 3],
        "primitive": "float64",
        "has_identifier": True,
        "parameters": {"hey": ["you", {"there": 3}]},
        "form_key": "yowzers",
    }

    form = ak._v2.forms.BitMaskedForm(
        "i8",
        ak._v2.forms.NumpyForm("float64"),
        True,
        False,
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "BitMaskedArray",
        "mask": "i8",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "valid_when": True,
        "lsb_order": False,
        "has_identifier": False,
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

    form = ak._v2.forms.EmptyForm(
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "EmptyArray",
        "has_identifier": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "EmptyArray",
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak._v2.forms.IndexedForm(
        "i64",
        ak._v2.forms.NumpyForm("float64"),
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "IndexedArray",
        "index": "i64",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
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

    form = ak._v2.forms.IndexedOptionForm(
        "i64",
        ak._v2.forms.NumpyForm("float64"),
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "IndexedOptionArray",
        "index": "i64",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
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

    form = ak._v2.forms.ListForm(
        "i64",
        "i64",
        ak._v2.forms.NumpyForm("float64"),
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "ListArray",
        "starts": "i64",
        "stops": "i64",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
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

    form = ak._v2.forms.ListOffsetForm(
        "i64",
        ak._v2.forms.NumpyForm("float64"),
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "ListOffsetArray",
        "offsets": "i64",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
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

    form = ak._v2.forms.RecordForm(
        [ak._v2.forms.NumpyForm("float64"), ak._v2.forms.NumpyForm("bool")],
        ["one", "two"],
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "RecordArray",
        "contents": {
            "one": {
                "class": "NumpyArray",
                "inner_shape": [],
                "primitive": "float64",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
            "two": {
                "class": "NumpyArray",
                "inner_shape": [],
                "primitive": "bool",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
        },
        "has_identifier": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "RecordArray",
        "contents": {
            "one": "float64",
            "two": "bool",
        },
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak._v2.forms.RecordForm(
        [ak._v2.forms.NumpyForm("float64"), ak._v2.forms.NumpyForm("bool")],
        None,
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "RecordArray",
        "contents": [
            {
                "class": "NumpyArray",
                "inner_shape": [],
                "primitive": "float64",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "inner_shape": [],
                "primitive": "bool",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
        ],
        "has_identifier": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "RecordArray",
        "contents": ["float64", "bool"],
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak._v2.forms.RegularForm(
        ak._v2.forms.NumpyForm("float64"),
        10,
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "RegularArray",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "size": 10,
        "has_identifier": False,
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

    form = ak._v2.forms.UnionForm(
        "i8",
        "i64",
        [ak._v2.forms.NumpyForm("float64"), ak._v2.forms.NumpyForm("bool")],
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i64",
        "contents": [
            {
                "class": "NumpyArray",
                "inner_shape": [],
                "primitive": "float64",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "inner_shape": [],
                "primitive": "bool",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
        ],
        "has_identifier": False,
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

    form = ak._v2.forms.UnmaskedForm(
        ak._v2.forms.NumpyForm("float64"),
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "UnmaskedArray",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "UnmaskedArray",
        "content": "float64",
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
