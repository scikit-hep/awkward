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

    builder.float64(1.1)
    builder.float64(2.2)
    builder.float64(3.3)

    assert ak.to_list(builder.snapshot()) == [1.1, 2.2, 3.3]


def test_byte_masked_form():
    form = ak.forms.ByteMaskedForm(
        "i8",
        ak.forms.NumpyForm([], 8, "d"),
        True,
        form_key="node0",
    )
    builder = ak.layout.TypedArrayBuilder(form)

    builder.float64(1.1)
    builder.float64(2.2)
    builder.float64(3.3)

    assert ak.to_list(builder.snapshot()) == [1.1, 2.2, 3.3]


def test_unmasked_form():
    form = ak.forms.UnmaskedForm(
        ak.forms.NumpyForm([], 8, "d"),
        form_key="node0",
    )
    builder = ak.layout.TypedArrayBuilder(form)

    builder.float64(1.1)
    builder.float64(2.2)
    builder.float64(3.3)

    assert ak.to_list(builder.snapshot()) == [1.1, 2.2, 3.3]


def test_unsupported_form():
    form = ak.forms.VirtualForm(ak.forms.NumpyForm([], 8, "d"), True)

    with pytest.raises(ValueError):
        ak.layout.TypedArrayBuilder(form)


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

    builder.begin_list()
    builder.float64(1.1)
    builder.begin_list()
    builder.int64(1)
    builder.end_list()
    builder.float64(2.2)
    builder.begin_list()
    builder.int64(1)
    builder.int64(2)
    builder.end_list()
    builder.end_list()

    builder.begin_list()
    builder.end_list()

    builder.begin_list()
    builder.float64(3.3)
    builder.begin_list()
    builder.int64(1)
    builder.int64(2)
    builder.int64(3)
    builder.end_list()
    builder.end_list()

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

    builder.int64(11)
    builder.int64(22)
    builder.int64(33)
    builder.int64(44)
    builder.int64(55)
    builder.int64(66)
    builder.int64(77)

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

    builder.null()
    builder.int64(11)
    builder.int64(22)
    builder.null()
    builder.int64(33)
    builder.int64(44)
    builder.null()
    builder.int64(55)
    builder.int64(66)
    builder.int64(77)

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

    builder.int64(11)
    builder.int64(22)
    builder.int64(33)
    builder.int64(44)
    builder.int64(55)
    builder.int64(66)
    builder.int64(77)

    assert ak.to_list(builder.snapshot()) == [[11, 22, 33], [44, 55, 66]]

    builder.int64(88)
    builder.int64(99)

    assert ak.to_list(builder.snapshot()) == [[11, 22, 33], [44, 55, 66], [77, 88, 99]]


def test_union_form():
    form = ak.forms.UnionForm(
        "i8",
        "i64",
        [ak.forms.NumpyForm([], 8, "d"), ak.forms.NumpyForm([], 1, "?")],
        form_key="node0",
    )

    builder = ak.layout.TypedArrayBuilder(form)

    builder.tag(0)
    builder.float64(1.1)
    builder.tag(1)
    builder.boolean(False)
    builder.tag(0)
    builder.float64(2.2)
    builder.tag(0)
    builder.float64(3.3)
    builder.tag(1)
    builder.boolean(True)
    builder.tag(0)
    builder.float64(4.4)
    builder.tag(1)
    builder.boolean(False)
    builder.tag(1)
    builder.boolean(True)
    builder.tag(0)
    builder.float64(-2.2)

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


def test_union2_form():
    form = ak.forms.UnionForm(
        "i8",
        "i64",
        [ak.forms.NumpyForm([], 8, "d"), ak.forms.NumpyForm([], 8, "d")],
        form_key="node0",
    )

    builder = ak.layout.TypedArrayBuilder(form)

    builder.tag(0)
    builder.float64(1.1)

    builder.tag(1)
    builder.float64(2.2)

    builder.tag(1)
    builder.float64(3.3)

    assert ak.to_list(builder.snapshot()) == [1.1, 2.2, 3.3]


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

    builder.tag(0)
    builder.float64(1.1)
    builder.tag(1)
    builder.boolean(False)
    builder.tag(2)
    builder.int64(11)
    builder.tag(0)
    builder.float64(2.2)
    builder.tag(1)
    builder.boolean(False)
    builder.tag(0)
    builder.float64(2.2)
    builder.tag(0)
    builder.float64(3.3)
    builder.tag(1)
    builder.boolean(True)
    builder.tag(0)
    builder.float64(4.4)
    builder.tag(1)
    builder.boolean(False)
    builder.tag(1)
    builder.boolean(True)
    builder.tag(0)
    builder.float64(-2.2)

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


def test_record_form():

    form = ak.forms.RecordForm(
        {"one": ak.forms.NumpyForm([], 8, "d"), "two": ak.forms.NumpyForm([], 8, "d")},
        form_key="node0",
    )
    builder = ak.layout.TypedArrayBuilder(form)

    # if record contents have the same type,
    # the fields alternate
    builder.float64(1.1)  # "one"
    builder.float64(2.2)  # "two"
    builder.float64(3.3)  # "one"
    builder.float64(4.4)  # "two"

    # etc.

    assert ak.to_list(builder.snapshot()) == [
        {"one": 1.1, "two": 2.2},
        {"one": 3.3, "two": 4.4},
    ]


def test_error_in_record_form():

    form = ak.forms.RecordForm(
        {"one": ak.forms.NumpyForm([], 8, "d"), "two": ak.forms.NumpyForm([], 8, "d")},
        form_key="node0",
    )
    builder = ak.layout.TypedArrayBuilder(form)

    # if record contents have the same type,
    # the fields alternate
    builder.float64(1.1)  # "one"
    builder.float64(2.2)  # "two"
    with pytest.raises(ValueError) as err:
        builder.int64(11)
    assert str(err.value) == "NumpyForm builder accepts only float64"


def test_error_in_numpy_form():

    form = ak.forms.NumpyForm([], 8, "d")

    builder = ak.layout.TypedArrayBuilder(form)

    builder.float64(1.1)
    builder.float64(2.2)
    with pytest.raises(ValueError) as err:
        builder.int64(11)
    assert str(err.value) == "NumpyForm builder accepts only float64"


def test_categorical_form():
    form = ak.forms.Form.fromjson(
        """
{
    "class": "IndexedArray64",
    "index": "i64",
    "content": "int64",
    "parameters": {
        "__array__": "categorical"
    }
}
"""
    )

    builder = ak.layout.TypedArrayBuilder(form)

    builder.int64(2019)
    builder.int64(2020)
    builder.int64(2021)
    builder.int64(2020)
    builder.int64(2019)
    builder.int64(2020)
    builder.int64(2020)
    builder.int64(2020)
    builder.int64(2020)
    builder.int64(2020)

    assert ak.to_list(builder.snapshot()) == [
        2019,
        2020,
        2021,
        2020,
        2019,
        2020,
        2020,
        2020,
        2020,
        2020,
    ]
    assert str(ak.type(ak.Array(builder.snapshot()))) == "10 * categorical[type=int64]"


def test_char_form():
    form = ak.forms.Form.fromjson(
        """
{
    "class": "NumpyArray",
    "itemsize": 1,
    "format": "B",
    "primitive": "uint8",
    "parameters": {
        "__array__": "char"
    }
}"""
    )

    builder = ak.layout.TypedArrayBuilder(form)

    builder.string("one")
    builder.string("two")
    builder.string("three")

    assert ak.to_list(builder.snapshot()) == "onetwothree"


def test_string_form():
    form = ak.forms.Form.fromjson(
        """
{
    "class": "ListOffsetArray64",
    "offsets": "i64",
    "content": {
        "class": "NumpyArray",
        "itemsize": 1,
        "format": "B",
        "primitive": "uint8",
        "parameters": {
            "__array__": "char"
        }
    },
    "parameters": {
        "__array__": "string"
    }
}"""
    )

    builder = ak.layout.TypedArrayBuilder(form)

    builder.string("one")
    builder.string("two")
    builder.string("three")

    assert ak.to_list(builder.snapshot()) == ["one", "two", "three"]


def test_empty_form():
    form = ak.forms.ListOffsetForm(
        "i64",
        ak.forms.ListOffsetForm(
            "i64",
            ak.forms.EmptyForm(),
            form_key="node1",
        ),
        form_key="node0",
    )

    builder = ak.layout.TypedArrayBuilder(form)

    builder.begin_list()
    builder.end_list()
    builder.begin_list()
    builder.begin_list()
    builder.end_list()
    builder.begin_list()
    builder.end_list()
    builder.begin_list()
    builder.end_list()
    builder.end_list()
    builder.begin_list()
    builder.begin_list()
    builder.end_list()
    builder.begin_list()
    builder.end_list()
    builder.end_list()
    builder.begin_list()
    builder.end_list()
    builder.begin_list()
    builder.begin_list()
    # builder.int64(1)
    # will fail with ValueError: EmptyArrayBuilder does not accept 'int64'
    builder.end_list()
    builder.end_list()

    assert ak.to_list(builder.snapshot()) == [[], [[], [], []], [[], []], [], [[]]]
