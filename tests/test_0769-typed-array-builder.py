# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward.forth

def test_list_offset_form():
    form = ak.forms.Form.fromjson("""
{
    "class": "ListOffsetArray64",
    "offsets": "i64",
    "content": {
        "class": "RecordArray",
        "contents": {
            "x": {
                "class": "NumpyArray",
                "primitive": "float64",
                "form_key": "node2"
            },
            "y": {
                "class": "ListOffsetArray64",
                "offsets": "i64",
                "content": {
                    "class": "NumpyArray",
                    "primitive": "int64",
                    "form_key": "node4"
                },
                "form_key": "node3"
            }
        },
        "form_key": "node1"
    },
    "form_key": "node0"
}
""")

    builder = ak.layout.TypedArrayBuilder(form)
    vm = awkward.forth.ForthMachine32(builder.to_vm())

    # initialise
    builder.connect(vm)

    builder.beginlist()
    builder.real(1.1)
    builder.beginlist()
    builder.integer(1)
    builder.endlist()
    builder.real(2.2)
    builder.beginlist()
    builder.integer(1)
    builder.integer(2)
    builder.endlist()
    builder.endlist()
    builder.beginlist()
    builder.endlist()
    builder.beginlist()
    builder.real(3.3)
    builder.beginlist()
    builder.integer(1)
    builder.integer(2)
    builder.integer(3)
    builder.endlist()
    builder.endlist()

    assert builder.form() == form

    assert ak.to_list(builder.snapshot()) == [
        [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}],
        [],
        [{"x": 3.3, "y": [1, 2, 3]}],
    ]

def test_indexed_form():
    form = ak.forms.Form.fromjson("""
{
    "class": "IndexedArray64",
    "index": "i64",
        "content": {
            "class": "NumpyArray",
            "primitive": "int64",
            "form_key": "node1"
        },
    "form_key": "node0"
}
""")

    builder = ak.layout.TypedArrayBuilder(form)
    vm = awkward.forth.ForthMachine32(builder.to_vm())

    # initialise
    builder.connect(vm)
    builder.integer(11)
    builder.integer(22)
    builder.integer(33)
    builder.integer(44)
    builder.integer(55)
    builder.integer(66)
    builder.integer(77)

    assert ak.to_list(builder.snapshot()) == [11, 22, 33, 44, 55, 66, 77]

def test_indexed_option_form():
    form = ak.forms.Form.fromjson("""
{
    "class": "IndexedOptionArray64",
    "index": "i64",
        "content": {
            "class": "NumpyArray",
            "primitive": "int64",
            "form_key": "node1"
        },
    "form_key": "node0"
}
""")

    builder = ak.layout.TypedArrayBuilder(form)
    vm = awkward.forth.ForthMachine32(builder.to_vm())

    # initialise
    builder.connect(vm)
    builder.null()
    builder.integer(11)
    builder.integer(22)
    builder.null()
    builder.integer(33)
    builder.integer(44)
    builder.null()
    builder.integer(55)
    builder.integer(66)
    builder.integer(77)

    assert ak.to_list(builder.snapshot()) == [None, 11, 22, None, 33, 44, None, 55, 66, 77]
