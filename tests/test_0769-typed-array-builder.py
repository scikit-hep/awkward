# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward.forth


def test_bit_masked_form():
    form = ak.forms.BitMaskedForm(
        "i8",
        ak.forms.NumpyForm([], 8, "d"),
        True,
        False,
        form_key="node0",
    )
    builder = ak.layout.TypedArrayBuilder(form)
    print(builder.to_vm())
    vm = awkward.forth.ForthMachine32(builder.to_vm())

    # initialise
    builder.connect(vm)

    builder.real(1.1)
    builder.real(2.2)
    builder.real(3.3)

    assert ak.to_list(builder.snapshot()) == [1.1, 2.2, 3.3]


def test_byte_masked_form():
    form = ak.forms.ByteMaskedForm(
        "i8",
        ak.forms.NumpyForm([], 8, "d"),
        True,
        form_key="node0",
    )
    builder = ak.layout.TypedArrayBuilder(form)
    print(builder.to_vm())
    vm = awkward.forth.ForthMachine32(builder.to_vm())

    # initialise
    builder.connect(vm)

    builder.real(1.1)
    builder.real(2.2)
    builder.real(3.3)

    assert ak.to_list(builder.snapshot()) == [1.1, 2.2, 3.3]


def test_unmasked_form():
    form = ak.forms.UnmaskedForm(
        ak.forms.NumpyForm([], 8, "d"),
        form_key="node0",
    )
    builder = ak.layout.TypedArrayBuilder(form)
    print(builder.to_vm())
    vm = awkward.forth.ForthMachine32(builder.to_vm())

    # initialise
    builder.connect(vm)

    builder.real(1.1)
    builder.real(2.2)
    builder.real(3.3)

    assert ak.to_list(builder.snapshot()) == [1.1, 2.2, 3.3]


def test_virtual_form():
    form = ak.forms.VirtualForm(ak.forms.NumpyForm([], 8, "d"), True)

    builder = ak.layout.TypedArrayBuilder(form)
    print(builder.to_vm())
    vm = awkward.forth.ForthMachine32(builder.to_vm())

    # initialise
    builder.connect(vm)

    builder.real(1.1)
    builder.real(2.2)
    builder.real(3.3)

    assert ak.to_list(builder.snapshot()) == [1.1, 2.2, 3.3]


def test_list_offset_form():
    form = ak.forms.Form.fromjson(
        """
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
"""
    )

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
    form = ak.forms.Form.fromjson(
        """
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
"""
    )

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
    form = ak.forms.Form.fromjson(
        """
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
"""
    )

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

    assert ak.to_list(builder.snapshot()) == [
        None,
        11,
        22,
        None,
        33,
        44,
        None,
        55,
        66,
        77,
    ]


def test_regular_form():
    form = ak.forms.Form.fromjson(
        """
{
    "class": "RegularArray",
    "size": 3,
        "content": {
            "class": "NumpyArray",
            "primitive": "int64",
            "form_key": "node1"
        },
    "form_key": "node0"
}
"""
    )

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

    assert ak.to_list(builder.snapshot()) == [[11, 22, 33], [44, 55, 66]]

    builder.integer(88)
    builder.integer(99)

    assert ak.to_list(builder.snapshot()) == [[11, 22, 33], [44, 55, 66], [77, 88, 99]]


def test_union_form():
    form = ak.forms.UnionForm(
        "i8",
        "i64",
        [ak.forms.NumpyForm([], 8, "d"), ak.forms.NumpyForm([], 1, "?")],
        form_key="node0",
    )

    builder = ak.layout.TypedArrayBuilder(form)
    print(builder.to_vm())
    vm = awkward.forth.ForthMachine32(builder.to_vm())

    # initialise
    builder.connect(vm)
    builder.real(1.1)
    builder.boolean(False)
    builder.real(2.2)
    builder.real(3.3)
    builder.boolean(True)
    builder.real(4.4)
    builder.boolean(False)
    builder.boolean(True)
    builder.real(-2.2)

    assert ak.to_list(builder.snapshot()) == [
        1.1,
        False,
        2.2,
        3.3,
        True,
        4.4,
        False,
        True,
        -2.2,
    ]


def test_union3_form():
    form = ak.forms.UnionForm(
        "i8",
        "i64",
        [
            ak.forms.NumpyForm([], 8, "d"),
            ak.forms.NumpyForm([], 1, "?"),
            ak.forms.NumpyForm([], 8, "q"),
        ],
        form_key="node0",
    )

    builder = ak.layout.TypedArrayBuilder(form)
    print(builder.to_vm())
    vm = awkward.forth.ForthMachine32(builder.to_vm())

    #     vm = awkward.forth.ForthMachine32("""
    # input data
    # output part0-node-id2-data float64
    # output part0-node-id3-data bool
    # output part0-node-id4-data int64
    # output part0-node0-tags int8
    # variable tag
    #
    # : node-id2-float64
    # 1 = if
    # 0 data seek
    # data d-> part0-node-id2-data
    # else
    # halt
    # then
    # ;
    # : node-id3-bool
    # 4 = if
    # 0 data seek
    # data ?-> part0-node-id3-data
    # else
    # halt
    # then
    # ;
    # : node-id4-int64
    # 0 = if
    # 0 data seek
    # data q-> part0-node-id4-data
    # else
    # halt
    # then
    # ;
    #
    # : node0-union
    # dup 1 = if
    # 0 tag !
    # tag @ part0-node0-tags <- stack
    # node-id2-float64
    # exit
    # then
    # dup 4 = if
    # 1 tag !
    # tag @ part0-node0-tags <- stack
    # node-id3-bool
    # exit
    # then
    # dup 0 = if
    # 2 tag !
    # tag @ part0-node0-tags <- stack
    # node-id4-int64
    # exit
    # then
    # ;
    #
    # 0 part0-node0-tags <- stack
    #
    # 0
    # begin
    # pause
    # node0-union
    # 1+
    # again
    # """)

    # initialise
    builder.connect(vm)
    builder.real(1.1)
    builder.boolean(False)
    builder.integer(11)
    builder.real(2.2)
    builder.boolean(False)
    builder.real(2.2)
    builder.real(3.3)
    builder.boolean(True)
    builder.real(4.4)
    builder.boolean(False)
    builder.boolean(True)
    builder.real(-2.2)

    assert ak.to_list(builder.snapshot()) == [
        1.1,
        False,
        11,
        2.2,
        False,
        2.2,
        3.3,
        True,
        4.4,
        False,
        True,
        -2.2,
    ]
