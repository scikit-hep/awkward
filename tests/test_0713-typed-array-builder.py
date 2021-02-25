# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward.forth

def test_jims_example():
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

    vm = awkward.forth.ForthMachine32("""
            input data
            output part0-node0-offsets int64
            output part0-node2-data float64
            output part0-node3-offsets int64
            output part0-node4-data int64

            : node4-int64
                {int64} = if
                    0 data seek
                    data q-> part0-node4-data
                else
                    halt
                then
            ;

            : node3-list
                {begin_list} <> if
                    halt
                then

                0
                begin
                    pause ( always pause before each list item )
                    dup {end_list} = if
                        drop
                        part0-node3-offsets +<- stack
                        exit
                    else
                        node4-int64
                        1+
                    then
                again
            ;

            : node2-float64
                {float64} = if
                    0 data seek
                    data d-> part0-node2-data
                else
                    halt
                then
            ;

            : node1-record
                node2-float64 pause ( pause after each field item except the last )
                node3-list
            ;

            : node0-list
                {begin_list} <> if
                    halt
                then

                0
                begin
                    pause ( always pause before each list item )
                    dup {end_list} = if
                        drop
                        part0-node0-offsets +<- stack
                        exit
                    else
                        node1-record
                        1+
                    then
                again
            ;

            0 part0-node0-offsets <- stack
            0 part0-node3-offsets <- stack

            0
            begin
                pause  ( always pause before each outermost array item )
                node0-list
                1+
            again
        """.format(int64=0, float64=1, begin_list=2, end_list=3))

    builder = ak.layout.TypedArrayBuilder(form)

    # initialise
    # builder.apply(form)
    builder.connect(vm)
    builder.debug_step()

    builder.beginlist()
    builder.debug_step()

    builder.real(1.1)
    builder.debug_step()

    builder.beginlist()
    builder.debug_step()

    builder.integer(1)
    builder.debug_step()

    builder.endlist()
    builder.debug_step()

    builder.real(2.2)
    builder.debug_step()

    builder.beginlist()
    builder.debug_step()

    builder.integer(1)
    builder.debug_step()

    builder.integer(2)
    builder.debug_step()

    builder.endlist()
    builder.debug_step()

    builder.endlist()
    builder.debug_step()

    builder.beginlist()
    builder.debug_step()

    builder.endlist()
    builder.debug_step()

    builder.beginlist()
    builder.debug_step()

    builder.real(3.3)
    builder.debug_step()

    builder.beginlist()
    builder.debug_step()

    builder.integer(1)
    builder.debug_step()

    builder.integer(2)
    builder.debug_step()

    builder.integer(3)
    builder.debug_step()

    builder.endlist()
    builder.debug_step()

    builder.endlist()
    builder.debug_step()

    builder.debug_step()

    assert builder.form() == form

    builder.snapshot()
    # assert ak.to_list(builder.snapshot()) == [
    #     [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}],
    #     [],
    #     [{"x": 3.3, "y": [1, 2, 3]}],
    # ]

# def test_typed_builder_form():
#     numpy_form = ak.forms.NumpyForm([], 8, "d")
#
#     # FIXME: generate it from a Form
#     vm_command = "input real\noutput content " + numpy_form.primitive + "\nreal i-> content"
#
#     vm32 = ak.forth.ForthMachine32(vm_command)
#
#     assert ak.to_list(vm32.bytecodes) == [[-34, 0, 0]]
#     assert (
#         vm32.decompiled == """input real
# output content float64
#
# real i-> content
# """
#     )
#
#     builder = ak.layout.TypedArrayBuilder()
#
#     # initialise
#     builder.apply(numpy_form)
#     #builder.connect(vm32)
#
#     #builder.real(5.5);
#
#     #assert ak.to_list(builder.snapshot()) == [5.5]
#
# def test_record_form_structure():
#
#     numpy_form = """{
# "class": "NumpyArray",
# "itemsize": 8,
# "format": "d",
# "primitive": "float64"
# }"""
#
#     list_offset_form = """{
# "class": "ListOffsetArray64",
# "offsets": "i64",
# "content": "int64"
# }"""
#
#     record_form = """{
# "class": "RecordArray",
# "contents": {
# "x": "float64",
# "y": {
# "class": "ListOffsetArray64",
# "offsets": "i64",
# "content": "int64"
# }
# }
# }"""
#
#     builder = ak.layout.ArrayBuilder()
#
#     builder.beginrecord()
#     builder.field("one")
#     builder.integer(1)
#     builder.field("two")
#     builder.real(1.1)
#     builder.endrecord()
#
#     builder.beginrecord()
#     builder.field("two")
#     builder.real(2.2)
#     builder.field("one")
#     builder.integer(2)
#     builder.endrecord()
#
#     builder.beginrecord()
#     builder.field("one")
#     builder.integer(3)
#     builder.field("two")
#     builder.real(3.3)
#     builder.endrecord()
#
#     typestrs = {}
#     assert str(builder.type(typestrs)) == '{"one": int64, "two": float64}'
#     assert ak.to_list(builder.snapshot()) == [
#         {"one": 1, "two": 1.1},
#         {"one": 2, "two": 2.2},
#         {"one": 3, "two": 3.3},
#     ]
