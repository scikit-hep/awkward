# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward.forth


def test_basics():
    vm32 = awkward.forth.ForthMachine32("")
    vm64 = awkward.forth.ForthMachine64("")

    assert str(ak.type(vm32.bytecodes)) == "var * int32"
    assert str(ak.type(vm64.bytecodes)) == "var * int32"
    assert vm32.dictionary == []
    assert vm64.dictionary == []
    assert vm32.stack_max_depth == 1024
    assert vm64.stack_max_depth == 1024
    assert vm32.recursion_max_depth == 1024
    assert vm64.recursion_max_depth == 1024
    assert vm32.output_initial_size == 1024
    assert vm64.output_initial_size == 1024
    assert vm32.output_resize_factor == 1.5
    assert vm64.output_resize_factor == 1.5

    vm32.stack_push(1)
    vm32.stack_push(2)
    vm32.stack_push(3)
    assert vm32.stack_pop() == 3
    assert vm32.stack == [1, 2]
    vm32.stack_clear()
    assert vm32.stack == []
    vm64.stack_push(1)
    vm64.stack_push(2)
    vm64.stack_push(3)
    assert vm64.stack_pop() == 3
    assert vm64.stack == [1, 2]
    vm64.stack_clear()
    assert vm64.stack == []

    assert vm32.variables == {}
    assert vm64.variables == {}
    assert vm32.outputs == {}
    assert vm64.outputs == {}


def test_comment_compilation():
    vm32 = awkward.forth.ForthMachine32("")
    assert ak.to_list(vm32.bytecodes) == [[]]
    assert vm32.decompiled == ""

    vm32 = awkward.forth.ForthMachine32("( comment )")
    assert ak.to_list(vm32.bytecodes) == [[]]

    vm32 = awkward.forth.ForthMachine32("\\ comment")
    assert ak.to_list(vm32.bytecodes) == [[]]

    vm32 = awkward.forth.ForthMachine32("\\ comment\n")
    assert ak.to_list(vm32.bytecodes) == [[]]

    vm32 = awkward.forth.ForthMachine32("1 2 ( comment ) 3 4")
    assert ak.to_list(vm32.bytecodes) == [[0, 1, 0, 2, 0, 3, 0, 4]]

    vm32 = awkward.forth.ForthMachine32("1 2 \\ comment \n 3 4")
    assert ak.to_list(vm32.bytecodes) == [[0, 1, 0, 2, 0, 3, 0, 4]]


def test_literal_compilation():
    vm32 = awkward.forth.ForthMachine32("1 2 3 4")
    assert ak.to_list(vm32.bytecodes) == [[0, 1, 0, 2, 0, 3, 0, 4]]
    assert (
        vm32.decompiled
        == """1
2
3
4
"""
    )


def test_userdef_compilation():
    vm32 = awkward.forth.ForthMachine32(": stuff 1 2 3 4 ;")
    assert ak.to_list(vm32.bytecodes) == [[], [0, 1, 0, 2, 0, 3, 0, 4]]
    assert (
        vm32.decompiled
        == """: stuff
  1
  2
  3
  4
;
"""
    )

    vm32 = awkward.forth.ForthMachine32(": foo 123 : bar 1 2 3 ; 321 ;")
    assert ak.to_list(vm32.bytecodes) == [[], [0, 123, 0, 321], [0, 1, 0, 2, 0, 3]]
    assert (
        vm32.decompiled
        == """: foo
  123
  321
;

: bar
  1
  2
  3
;
"""
    )

    vm32 = awkward.forth.ForthMachine32(": empty ;")
    assert ak.to_list(vm32.bytecodes) == [[], []]
    assert (
        vm32.decompiled
        == """: empty
;
"""
    )

    vm32 = awkward.forth.ForthMachine32(": infinite recurse ;")
    assert ak.to_list(vm32.bytecodes) == [[], [60]]
    assert (
        vm32.decompiled
        == """: infinite
  infinite
;
"""
    )


def test_declarations_compilation():
    vm32 = awkward.forth.ForthMachine32("variable x")
    assert ak.to_list(vm32.bytecodes) == [[]]
    assert (
        vm32.decompiled
        == """variable x
"""
    )

    vm32 = awkward.forth.ForthMachine32("variable x variable y")
    assert ak.to_list(vm32.bytecodes) == [[]]
    assert (
        vm32.decompiled
        == """variable x
variable y
"""
    )

    vm32 = awkward.forth.ForthMachine32("input x")
    assert ak.to_list(vm32.bytecodes) == [[]]
    assert (
        vm32.decompiled
        == """input x
"""
    )

    vm32 = awkward.forth.ForthMachine32("output x int32")
    assert ak.to_list(vm32.bytecodes) == [[]]
    assert (
        vm32.decompiled
        == """output x int32
"""
    )

    vm32 = awkward.forth.ForthMachine32("output x int32 input y variable z")
    assert ak.to_list(vm32.bytecodes) == [[]]
    assert (
        vm32.decompiled
        == """variable z
input y
output x int32
"""
    )

    for dtype in [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ]:
        vm32 = awkward.forth.ForthMachine32("output x " + dtype)
        assert vm32.decompiled == "output x " + dtype + "\n"


def test_control_compilation():
    vm32 = awkward.forth.ForthMachine32("halt")
    assert (
        vm32.decompiled
        == """halt
"""
    )

    vm32 = awkward.forth.ForthMachine32("pause")
    assert (
        vm32.decompiled
        == """pause
"""
    )

    vm32 = awkward.forth.ForthMachine32("exit")
    assert (
        vm32.decompiled
        == """exit
"""
    )


def test_if_compilation():
    vm32 = awkward.forth.ForthMachine32("-1 if then")
    assert (
        vm32.decompiled
        == """-1
if
then
"""
    )

    vm32 = awkward.forth.ForthMachine32("-1 if 3 5 + then")
    assert (
        vm32.decompiled
        == """-1
if
  3
  5
  +
then
"""
    )

    vm32 = awkward.forth.ForthMachine32("-1 if else then")
    assert (
        vm32.decompiled
        == """-1
if
else
then
"""
    )

    vm32 = awkward.forth.ForthMachine32("-1 if 3 5 + else 123 then")
    assert (
        vm32.decompiled
        == """-1
if
  3
  5
  +
else
  123
then
"""
    )

    vm32 = awkward.forth.ForthMachine32(": foo -1 if 3 5 + then ; foo")
    assert (
        vm32.decompiled
        == """: foo
  -1
  if
    3
    5
    +
  then
;

foo
"""
    )


def test_loop_compilation():
    vm32 = awkward.forth.ForthMachine32("10 0 do loop")
    assert (
        vm32.decompiled
        == """10
0
do
loop
"""
    )

    vm32 = awkward.forth.ForthMachine32("10 0 do i loop")
    assert (
        vm32.decompiled
        == """10
0
do
  i
loop
"""
    )

    vm32 = awkward.forth.ForthMachine32("10 0 do i +loop")
    assert (
        vm32.decompiled
        == """10
0
do
  i
+loop
"""
    )

    vm32 = awkward.forth.ForthMachine32("10 0 do 5 0 do 3 1 do i j k loop loop loop")
    assert (
        vm32.decompiled
        == """10
0
do
  5
  0
  do
    3
    1
    do
      i
      j
      k
    loop
  loop
loop
"""
    )

    vm32 = awkward.forth.ForthMachine32("begin again")
    assert (
        vm32.decompiled
        == """begin
again
"""
    )

    vm32 = awkward.forth.ForthMachine32("begin 123 again")
    assert (
        vm32.decompiled
        == """begin
  123
again
"""
    )

    vm32 = awkward.forth.ForthMachine32("begin until")
    assert (
        vm32.decompiled
        == """begin
until
"""
    )

    vm32 = awkward.forth.ForthMachine32("begin 123 until")
    assert (
        vm32.decompiled
        == """begin
  123
until
"""
    )

    vm32 = awkward.forth.ForthMachine32("begin while repeat")
    print(ak.to_list(vm32.bytecodes))
    assert (
        vm32.decompiled
        == """begin
while
repeat
"""
    )

    vm32 = awkward.forth.ForthMachine32("begin 123 while 321 repeat")
    print(ak.to_list(vm32.bytecodes))
    assert (
        vm32.decompiled
        == """begin
  123
while
  321
repeat
"""
    )


def test_io_compilation():
    vm32 = awkward.forth.ForthMachine32("variable x x !")
    assert (
        vm32.decompiled
        == """variable x

x !
"""
    )

    vm32 = awkward.forth.ForthMachine32("variable x x +!")
    assert (
        vm32.decompiled
        == """variable x

x +!
"""
    )

    vm32 = awkward.forth.ForthMachine32("variable x x @")
    assert (
        vm32.decompiled
        == """variable x

x @
"""
    )

    vm32 = awkward.forth.ForthMachine32("input x x len")
    assert (
        vm32.decompiled
        == """input x

x len
"""
    )

    vm32 = awkward.forth.ForthMachine32("input x x pos")
    assert (
        vm32.decompiled
        == """input x

x pos
"""
    )

    vm32 = awkward.forth.ForthMachine32("input x x end")
    assert (
        vm32.decompiled
        == """input x

x end
"""
    )

    vm32 = awkward.forth.ForthMachine32("input x x seek")
    assert (
        vm32.decompiled
        == """input x

x seek
"""
    )

    vm32 = awkward.forth.ForthMachine32("input x x skip")
    assert (
        vm32.decompiled
        == """input x

x skip
"""
    )

    vm32 = awkward.forth.ForthMachine32("output x int32 x len")
    assert (
        vm32.decompiled
        == """output x int32

x len
"""
    )

    vm32 = awkward.forth.ForthMachine32("output x int32 x rewind")
    assert (
        vm32.decompiled
        == """output x int32

x rewind
"""
    )


def test_read_compilation():
    vm32 = awkward.forth.ForthMachine32("input x x i-> stack")
    assert (
        vm32.decompiled
        == """input x

x i-> stack
"""
    )

    vm32 = awkward.forth.ForthMachine32("input x output y int32 x i-> y")
    print(ak.to_list(vm32.bytecodes))
    print(vm32.decompiled)
    assert (
        vm32.decompiled
        == """input x
output y int32

x i-> y
"""
    )

    for rep in ["", "#"]:
        for big in ["", "!"]:
            for tpe in [
                "?",
                "b",
                "h",
                "i",
                "q",
                "n",
                "B",
                "H",
                "I",
                "Q",
                "N",
                "f",
                "d",
            ]:
                if not (big == "!" and tpe in ("?", "b", "B")):
                    source = """input x

x {0} stack
""".format(
                        rep + big + tpe + "->"
                    )
                    vm32 = awkward.forth.ForthMachine32(source)
                    assert vm32.decompiled == source
                    source = """input x
output y int32

x {0} y
""".format(
                        rep + big + tpe + "->"
                    )
                    vm32 = awkward.forth.ForthMachine32(source)
                    assert vm32.decompiled == source
                    del vm32


def test_everything_else_compilation():
    source = (
        "\n".join(
            [
                "dup",
                "drop",
                "swap",
                "over",
                "rot",
                "nip",
                "tuck",
                "+",
                "-",
                "*",
                "/",
                "mod",
                "/mod",
                "negate",
                "1+",
                "1-",
                "abs",
                "min",
                "max",
                "=",
                "<>",
                ">",
                ">=",
                "<",
                "<=",
                "0=",
                "invert",
                "and",
                "or",
                "xor",
                "lshift",
                "rshift",
                "false",
                "true",
            ]
        )
        + "\n"
    )
    vm32 = awkward.forth.ForthMachine32(source)
    assert vm32.decompiled == source


def test_input_output():
    vm32 = awkward.forth.ForthMachine32("input x output y int32")
    vm32.begin({"x": np.array([1, 2, 3])})
    assert isinstance(vm32["y"], ak.layout.NumpyArray)


def test_stepping():
    vm32 = awkward.forth.ForthMachine32("1 2 3 4")
    vm32.begin()
    assert vm32.stack == []
    vm32.step()
    assert vm32.stack == [1]
    vm32.step()
    assert vm32.stack == [1, 2]
    vm32.step()
    assert vm32.stack == [1, 2, 3]
    vm32.step()
    assert vm32.stack == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        vm32.step()
