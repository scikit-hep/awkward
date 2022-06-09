# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward.forth


def test_empty_form():
    form = """
{
    "class": "EmptyArray"
}
    """
    builder = ak._v2.highlevel.LayoutBuilder(form)
    array = builder.snapshot()
    assert ak._v2.operations.type(array) == ak._v2.types.ArrayType(
        ak._v2.types.UnknownType(), 0
    )

    assert ak._v2.to_list(array) == []


def test_numpy_bool_form():
    form = """
{
    "class": "NumpyArray",
    "primitive": "bool"
}    """
    builder = ak._v2.highlevel.LayoutBuilder(form)

    builder.boolean(True)
    builder.boolean(True)
    builder.boolean(False)
    builder.boolean(True)

    array = builder.snapshot()
    assert ak._v2.operations.type(array) == ak._v2.types.ArrayType(
        ak._v2.types.NumpyType("bool"), 4
    )  # FIXME: ak._v2.types.NumpyType("bool")

    assert ak._v2.to_list(array) == [True, True, False, True]


def test_numpy_int_form():
    form = """
{
    "class": "NumpyArray",
    "primitive": "int64"
}    """
    builder = ak._v2.highlevel.LayoutBuilder(form)

    builder.int64(1)
    builder.int64(2)
    builder.int64(3)
    builder.int64(4)

    array = builder.snapshot()
    assert ak._v2.operations.type(array) == ak._v2.types.ArrayType(
        ak._v2.types.NumpyType("int64"), 4
    )  # FIXME: ak._v2.types.NumpyType("int64")

    assert ak._v2.to_list(array) == [1, 2, 3, 4]


def test_numpy_float_form():
    form = """
{
    "class": "NumpyArray",
    "primitive": "float64"
}    """
    builder = ak._v2.highlevel.LayoutBuilder(form)

    builder.float64(1.1)
    builder.float64(2.2)
    builder.float64(3.3)
    builder.float64(4.4)

    array = builder.snapshot()
    assert ak._v2.operations.type(array) == ak._v2.types.ArrayType(
        ak._v2.types.NumpyType("float64"), 4
    )  # FIXME: ak._v2.types.NumpyType("float64")

    assert ak._v2.to_list(array) == [1.1, 2.2, 3.3, 4.4]


@pytest.mark.skipif(
    ak._util.win,
    reason="FIXME: the test fails on Windows",
)
def test_numpy_complex_form():
    form = """
{
    "class": "NumpyArray",
    "primitive": "complex128"
}    """
    builder = ak._v2.highlevel.LayoutBuilder(form)

    builder.complex(1.0 + 0.1j)
    builder.complex(2.0 + 0.2j)
    builder.complex(3.0 + 0.3j)
    builder.complex(4.0 + 0.4j)

    array = builder.snapshot()
    assert ak._v2.operations.type(array) == ak._v2.types.ArrayType(
        ak._v2.types.NumpyType("complex128"), 4
    )  # FIXME: ak._v2.types.NumpyType("complex128")

    assert ak._v2.to_list(array) == [1.0 + 0.1j, 2.0 + 0.2j, 3.0 + 0.3j, 4.0 + 0.4j]


def test_record_form2():
    form = """
{
    "class": "RecordArray",
    "contents": {
        "one": "float64",
        "two": "int64"
    },
    "form_key": "node0"
}
    """
    builder = ak._v2.highlevel.LayoutBuilder(form)

    # the fields alternate
    builder.float64(1.1)  # "one"
    builder.int64(2)  # "two"
    builder.float64(3.3)  # "one"
    builder.int64(4)  # "two"

    # etc.
    array = builder.snapshot()
    assert ak._v2.operations.type(array) == ak._v2.types.ArrayType(
        ak._v2.types.RecordType(
            [ak._v2.types.NumpyType("float64"), ak._v2.types.NumpyType("int64")],
            ["one", "two"],
        ),
        2,
    )

    assert ak._v2.to_list(array) == [
        {"one": 1.1, "two": 2},
        {"one": 3.3, "two": 4},
    ]


def test_bit_masked_form():
    form = """
{
    "class": "BitMaskedArray",
    "mask": "i8",
    "content": "float64",
    "valid_when": true,
    "lsb_order": false,
    "form_key": "node0"
}
    """
    builder = ak._v2.highlevel.LayoutBuilder(form)

    builder.float64(1.1)
    builder.float64(2.2)
    builder.float64(3.3)

    assert ak._v2.to_list(builder.snapshot()) == [1.1, 2.2, 3.3]


def test_byte_masked_form():
    form = """
{
    "class": "ByteMaskedArray",
    "mask": "i8",
    "content": "float64",
    "valid_when": true,
    "form_key": "node0"
}
    """

    builder = ak._v2.highlevel.LayoutBuilder(form)

    builder.float64(1.1)
    builder.float64(2.2)
    builder.float64(3.3)

    assert ak._v2.to_list(builder.snapshot()) == [1.1, 2.2, 3.3]


def test_unmasked_form():
    form = """
{
    "class": "UnmaskedArray",
    "content": "float64",
    "form_key": "node0"
}
"""
    builder = ak._v2.highlevel.LayoutBuilder(form)

    builder.float64(1.1)
    builder.float64(2.2)
    builder.float64(3.3)

    assert ak._v2.to_list(builder.snapshot()) == [1.1, 2.2, 3.3]


def test_unsupported_form():
    form = """
{
    "class": "VirtualArray",
    "form": "float64",
    "has_length": true
}
           """

    with pytest.raises(ValueError):
        ak._v2.highlevel.LayoutBuilder(form)


def test_list_offset_form():
    form = """
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

    builder = ak._v2.highlevel.LayoutBuilder(form)

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

    assert ak._v2.to_list(builder.snapshot()) == [
        [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}],
        [],
        [{"x": 3.3, "y": [1, 2, 3]}],
    ]


def test_indexed_form():
    form = """
{
    "class": "IndexedArray",
    "index": "i64",
    "content": {
        "class": "NumpyArray",
        "primitive": "int64",
        "form_key": "node1"
    },
    "form_key": "node0"
}
    """

    builder = ak._v2.highlevel.LayoutBuilder(form)

    builder.int64(11)
    builder.int64(22)
    builder.int64(33)
    builder.int64(44)
    builder.int64(55)
    builder.int64(66)
    builder.int64(77)

    assert ak._v2.to_list(builder.snapshot()) == [11, 22, 33, 44, 55, 66, 77]


def test_indexed_option_form():
    form = """
{
    "class": "IndexedOptionArray",
    "index": "i64",
    "content": {
        "class": "NumpyArray",
        "primitive": "int64",
        "form_key": "node1"
    },
    "form_key": "node0"
}
    """

    builder = ak._v2.highlevel.LayoutBuilder(form)

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

    # FIXME: the first index element is duplicated
    assert ak._v2.to_list(builder.snapshot()[1:]) == [
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
    form = """
{
    "class": "RegularArray",
    "content": "int64",
    "size": 3
}
    """

    builder = ak._v2.highlevel.LayoutBuilder(form)

    builder.int64(11)
    builder.int64(22)
    builder.int64(33)
    builder.int64(44)
    builder.int64(55)
    builder.int64(66)
    builder.int64(77)

    assert ak._v2.to_list(builder.snapshot()) == [[11, 22, 33], [44, 55, 66]]

    builder.int64(88)
    builder.int64(99)

    assert ak._v2.to_list(builder.snapshot()) == [
        [11, 22, 33],
        [44, 55, 66],
        [77, 88, 99],
    ]


def test_union_form():
    form = """
{
    "class": "UnionArray",
    "tags": "i8",
    "index": "i64",
    "contents": [
        "float64",
        "bool"
    ],
    "form_key": "node0"
}
    """

    builder = ak._v2.highlevel.LayoutBuilder(form)

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

    assert ak._v2.to_list(builder.snapshot()) == [
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
    form = """
{
    "class": "UnionArray",
    "tags": "i8",
    "index": "i64",
    "contents": [
        "float64",
        "float64"
    ],
    "form_key": "node0"
}
    """

    builder = ak._v2.highlevel.LayoutBuilder(form)

    builder.tag(0)
    builder.float64(1.1)

    builder.tag(1)
    builder.float64(2.2)

    builder.tag(1)
    builder.float64(3.3)

    assert ak._v2.to_list(builder.snapshot()) == [1.1, 2.2, 3.3]


def test_union3_form():
    form = """
{
    "class": "UnionArray",
    "tags": "i8",
    "index": "i64",
    "contents": [
        "float64",
        "bool",
        "int64"
    ],
    "form_key": "node0"
}
    """

    builder = ak._v2.highlevel.LayoutBuilder(form)

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

    assert ak._v2.to_list(builder.snapshot()) == [
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
    form = """
{
    "class": "RecordArray",
    "contents": {
        "one": "float64",
        "two": "float64"
    },
    "form_key": "node0"
}
    """
    builder = ak._v2.highlevel.LayoutBuilder(form)

    # if record contents have the same type,
    # the fields alternate
    builder.float64(1.1)  # "one"
    builder.float64(2.2)  # "two"
    builder.float64(3.3)  # "one"
    builder.float64(4.4)  # "two"

    # etc.

    assert ak._v2.to_list(builder.snapshot()) == [
        {"one": 1.1, "two": 2.2},
        {"one": 3.3, "two": 4.4},
    ]


def test_error_in_record_form():
    form = """
{
    "class": "RecordArray",
    "contents": {
        "one": "float64",
        "two": "float64"
    },
    "form_key": "node0"
}
    """

    builder = ak._v2.highlevel.LayoutBuilder(form)

    # if record contents have the same type,
    # the fields alternate
    builder.float64(1.1)  # "one"
    builder.float64(2.2)  # "two"
    with pytest.raises(ValueError) as err:
        builder.int64(11)
    assert str(err.value) == "NumpyForm builder accepts only float64"


def test_error_in_numpy_form():
    form = """
{
    "class": "NumpyArray",
    "itemsize": 8,
    "format": "d",
    "primitive": "float64"
}
    """

    builder = ak._v2.highlevel.LayoutBuilder(form)

    builder.float64(1.1)
    builder.float64(2.2)
    with pytest.raises(ValueError) as err:
        builder.int64(11)
    assert str(err.value) == "NumpyForm builder accepts only float64"


@pytest.mark.skip(reason="FIXME: check categorical type")
def test_categorical_form():
    form = """
{
    "class": "IndexedArray",
    "index": "i64",
    "content": "int64",
    "parameters": {
        "__array__": "categorical"
    }
}
"""

    builder = ak._v2.highlevel.LayoutBuilder(form)

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

    assert ak._v2.to_list(builder.snapshot()) == [
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
    form = """
{
    "class": "NumpyArray",
    "itemsize": 1,
    "format": "B",
    "primitive": "uint8",
    "parameters": {
        "__array__": "char"
    }
}"""

    builder = ak._v2.highlevel.LayoutBuilder(form)

    builder.string("one")
    builder.string("two")
    builder.string("three")

    assert ak._v2.to_list(builder.snapshot()) == "onetwothree"


def test_string_form():
    form = """
{
    "class": "ListOffsetArray",
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

    builder = ak._v2.highlevel.LayoutBuilder(form)

    builder.string("one")
    builder.string("two")
    builder.string("three")

    assert ak._v2.to_list(builder.snapshot()) == ["one", "two", "three"]


def test_list_offset_empty_form():
    form = """
{
    "class": "ListOffsetArray",
    "offsets": "i64",
    "content": {
        "class": "EmptyArray"
    }
}
"""

    builder = ak._v2.highlevel.LayoutBuilder(form)

    builder.begin_list()
    builder.end_list()
    builder.begin_list()
    builder.end_list()
    builder.begin_list()
    builder.end_list()

    assert ak._v2.to_list(builder.snapshot()) == [[], [], []]


def test_list_offset_list_offset_empty_form():
    form = """
{
    "class": "ListOffsetArray",
    "offsets": "i64",
    "content": {
        "class": "ListOffsetArray",
        "offsets": "i64",
        "content": {
            "class": "EmptyArray"
        }
    }
}
"""

    builder = ak._v2.highlevel.LayoutBuilder(form)

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

    assert ak._v2.to_list(builder.snapshot()) == [[], [[], [], []], [[], []], [], [[]]]
