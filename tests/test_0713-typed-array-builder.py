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

    assert ak.to_list(builder.snapshot()) == [
        [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}],
        [],
        [{"x": 3.3, "y": [1, 2, 3]}],
    ]
