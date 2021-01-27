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


def test_input_output_type():
    vm32 = awkward.forth.ForthMachine32("input x output y int32")
    vm32.begin({"x": np.array([1, 2, 3])})
    assert isinstance(vm32["y"], ak.layout.NumpyArray)


def test_stepping():
    vm32 = awkward.forth.ForthMachine32("1 2 3 4")
    with pytest.raises(ValueError):
        vm32.step()
    assert not vm32.is_ready
    assert vm32.is_done

    vm32.begin()
    assert vm32.is_ready
    assert not vm32.is_done
    assert vm32.stack == []
    vm32.step()
    assert vm32.stack == [1]
    vm32.step()
    assert vm32.stack == [1, 2]
    vm32.step()
    assert vm32.stack == [1, 2, 3]
    vm32.step()
    assert vm32.stack == [1, 2, 3, 4]
    assert vm32.is_ready
    assert vm32.is_done
    with pytest.raises(ValueError):
        vm32.step()

    vm32.reset()
    with pytest.raises(ValueError):
        vm32.step()

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


def test_running():
    vm32 = awkward.forth.ForthMachine32("1 2 3 4")
    vm32.run()
    assert vm32.is_ready
    assert vm32.is_done

    assert vm32.stack == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        vm32.step()

    vm32.run()

    assert vm32.stack == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        vm32.step()


def test_pausing():
    vm32 = awkward.forth.ForthMachine32("1 2 pause 3 4 5")

    vm32.run()
    assert vm32.is_ready
    assert not vm32.is_done
    assert vm32.stack == [1, 2]
    vm32.step()
    assert vm32.is_ready
    assert not vm32.is_done
    assert vm32.stack == [1, 2, 3]
    vm32.resume()
    assert vm32.is_ready
    assert vm32.is_done
    assert vm32.stack == [1, 2, 3, 4, 5]


def test_calling():
    vm32 = awkward.forth.ForthMachine32(": foo 999 ; 1 2 3 pause 4 5")

    vm32.begin()
    assert vm32.stack == []
    assert not vm32.is_done

    vm32.call("foo")
    assert vm32.stack == [999]
    assert not vm32.is_done

    vm32.step()
    assert vm32.stack == [999, 1]
    assert not vm32.is_done

    vm32.call("foo")
    assert vm32.stack == [999, 1, 999]
    assert not vm32.is_done

    vm32.resume()
    assert vm32.stack == [999, 1, 999, 2, 3]
    assert not vm32.is_done

    vm32.resume()
    assert vm32.stack == [999, 1, 999, 2, 3, 4, 5]
    assert vm32.is_done

    vm32.run()
    assert vm32.stack == [1, 2, 3]
    assert not vm32.is_done

    vm32.call("foo")
    assert vm32.stack == [1, 2, 3, 999]
    assert not vm32.is_done

    vm32.call("foo")
    assert vm32.stack == [1, 2, 3, 999, 999]
    assert not vm32.is_done

    vm32.step()
    assert vm32.stack == [1, 2, 3, 999, 999, 4]
    assert not vm32.is_done

    vm32.step()
    assert vm32.stack == [1, 2, 3, 999, 999, 4, 5]
    assert vm32.is_done


def test_calling2():
    vm32 = awkward.forth.ForthMachine32(
        """
: bar 999 ;
: foo bar pause bar ;
1 2 3 pause 4 5"""
    )

    vm32.run()
    assert vm32.stack == [1, 2, 3]
    assert not vm32.is_done

    vm32.call("foo")
    assert vm32.stack == [1, 2, 3, 999]
    assert not vm32.is_done

    vm32.resume()
    assert vm32.stack == [1, 2, 3, 999, 999]
    assert not vm32.is_done

    vm32.resume()
    assert vm32.stack == [1, 2, 3, 999, 999, 4, 5]
    assert vm32.is_done

    vm32.call("foo")
    assert vm32.stack == [1, 2, 3, 999, 999, 4, 5, 999]
    assert not vm32.is_done

    vm32.resume()
    assert vm32.stack == [1, 2, 3, 999, 999, 4, 5, 999, 999]
    assert vm32.is_done


def test_calling3():
    vm32 = awkward.forth.ForthMachine32(
        """
: bar 999 ;
: foo bar pause bar ;"""
    )

    vm32.run()
    assert vm32.stack == []
    assert vm32.is_done

    vm32.call("foo")
    assert vm32.stack == [999]
    assert not vm32.is_done

    vm32.resume()
    assert vm32.stack == [999, 999]
    assert vm32.is_done


def test_halt():
    vm32 = awkward.forth.ForthMachine32("1 2 3 halt 4 5")

    vm32.run(raise_user_halt=False)
    assert vm32.stack == [1, 2, 3]
    assert vm32.is_done


def test_halt2():
    vm32 = awkward.forth.ForthMachine32(
        """
: bar 999 ;
: foo bar halt bar ;
1 2 3 pause 4 5"""
    )

    vm32.run()
    assert vm32.stack == [1, 2, 3]
    assert not vm32.is_done

    vm32.call("foo", raise_user_halt=False)
    assert vm32.stack == [1, 2, 3, 999]
    assert vm32.is_done

    with pytest.raises(ValueError):
        vm32.step()

    vm32.run()
    assert vm32.stack == [1, 2, 3]
    assert not vm32.is_done

    vm32.resume()
    assert vm32.stack == [1, 2, 3, 4, 5]
    assert vm32.is_done


def test_do():
    vm32 = awkward.forth.ForthMachine32("5 0 do i loop")
    vm32.run()
    assert vm32.stack == [0, 1, 2, 3, 4]

    vm32 = awkward.forth.ForthMachine32("5 0 do i pause loop")
    vm32.run()
    assert vm32.stack == [0]
    vm32.resume()
    assert vm32.stack == [0, 1]
    vm32.resume()
    assert vm32.stack == [0, 1, 2]
    vm32.resume()
    assert vm32.stack == [0, 1, 2, 3]
    vm32.resume()
    assert vm32.stack == [0, 1, 2, 3, 4]

    vm32 = awkward.forth.ForthMachine32(": foo 123 pause ; 5 0 do i foo loop")
    vm32.run()
    assert vm32.stack == [0, 123]
    vm32.resume()
    assert vm32.stack == [0, 123, 1, 123]
    vm32.resume()
    assert vm32.stack == [0, 123, 1, 123, 2, 123]
    vm32.resume()
    assert vm32.stack == [0, 123, 1, 123, 2, 123, 3, 123]
    vm32.resume()
    assert vm32.stack == [0, 123, 1, 123, 2, 123, 3, 123, 4, 123]

    vm32 = awkward.forth.ForthMachine32(": foo 123 pause ; 5 0 do i foo loop")
    vm32.run()
    assert vm32.stack == [0, 123]
    vm32.call("foo")
    assert vm32.stack == [0, 123, 123]
    vm32.resume()
    assert vm32.stack == [0, 123, 123, 1, 123]


def test_errors():
    # util::ForthError::not_ready
    vm32 = awkward.forth.ForthMachine32("1 2 3")
    with pytest.raises(ValueError):
        vm32.resume()
    with pytest.raises(ValueError):
        vm32.step()

    # util::ForthError::is_done
    vm32 = awkward.forth.ForthMachine32("1 2 3")
    vm32.run()
    with pytest.raises(ValueError):
        vm32.resume()
    with pytest.raises(ValueError):
        vm32.step()

    # util::ForthError::user_halt
    vm32 = awkward.forth.ForthMachine32("halt")
    with pytest.raises(ValueError):
        vm32.run()
    vm32.run(raise_user_halt=False)

    # util::ForthError::recursion_depth_exceeded
    vm32 = awkward.forth.ForthMachine32(": infinite infinite ; infinite")
    with pytest.raises(ValueError):
        vm32.run()
    vm32.run(raise_recursion_depth_exceeded=False)

    # util::ForthError::stack_underflow
    vm32 = awkward.forth.ForthMachine32("1 +")
    with pytest.raises(ValueError):
        vm32.run()
    vm32.run(raise_stack_underflow=False)

    # util::ForthError::stack_overflow
    vm32 = awkward.forth.ForthMachine32("1025 0 do i loop")
    with pytest.raises(ValueError):
        vm32.run()
    vm32.run(raise_stack_overflow=False)

    # util::ForthError::read_beyond
    vm32 = awkward.forth.ForthMachine32("input x x b-> stack")
    with pytest.raises(ValueError):
        vm32.run({"x": np.array([])})
    vm32.run({"x": np.array([])}, raise_read_beyond=False)

    # util::ForthError::seek_beyond
    vm32 = awkward.forth.ForthMachine32("input x 1 x seek")
    with pytest.raises(ValueError):
        vm32.run({"x": np.array([])})
    vm32.run({"x": np.array([])}, raise_seek_beyond=False)

    # util::ForthError::skip_beyond
    vm32 = awkward.forth.ForthMachine32("input x 1 x skip")
    with pytest.raises(ValueError):
        vm32.run({"x": np.array([])})
    vm32.run({"x": np.array([])}, raise_skip_beyond=False)

    # util::ForthError::rewind_beyond
    vm32 = awkward.forth.ForthMachine32("output x int32 1 x rewind")
    with pytest.raises(ValueError):
        vm32.run()
    vm32.run(raise_rewind_beyond=False)

    # util::ForthError::division_by_zero
    vm32 = awkward.forth.ForthMachine32("123 0 /")
    with pytest.raises(ValueError):
        vm32.run()
    vm32.run(raise_division_by_zero=False)

    vm32 = awkward.forth.ForthMachine32("-123 0 /")
    with pytest.raises(ValueError):
        vm32.run()
    vm32.run(raise_division_by_zero=False)

    vm32 = awkward.forth.ForthMachine32("123 0 mod")
    with pytest.raises(ValueError):
        vm32.run()
    vm32.run(raise_division_by_zero=False)

    vm32 = awkward.forth.ForthMachine32("-123 0 mod")
    with pytest.raises(ValueError):
        vm32.run()
    vm32.run(raise_division_by_zero=False)

    vm32 = awkward.forth.ForthMachine32("123 0 /mod")
    with pytest.raises(ValueError):
        vm32.run()
    vm32.run(raise_division_by_zero=False)

    vm32 = awkward.forth.ForthMachine32("-123 0 /mod")
    with pytest.raises(ValueError):
        vm32.run()
    vm32.run(raise_division_by_zero=False)


def test_gforth():
    # Expected outputs were all generated by running gforth:
    #
    #     echo "100 maxdepth-.s ! {0} .s" | gforth
    #
    # and parsing the stack output (none of the stacks are larger than 100).

    vm32 = awkward.forth.ForthMachine32(": foo if 999 then ; -1 foo")
    vm32.run()
    assert vm32.stack == [999]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 then ; 0 foo")
    vm32.run()
    assert vm32.stack == []

    vm32 = awkward.forth.ForthMachine32(": foo if 999 then ; 1 foo")
    vm32.run()
    assert vm32.stack == [999]

    vm32 = awkward.forth.ForthMachine32(": foo if if 999 then then ; -1 -1 foo")
    vm32.run()
    assert vm32.stack == [999]

    vm32 = awkward.forth.ForthMachine32(": foo if if 999 then then ; 0 -1 foo")
    vm32.run()
    assert vm32.stack == []

    vm32 = awkward.forth.ForthMachine32(": foo if if 999 then then ; 1 -1 foo")
    vm32.run()
    assert vm32.stack == [999]

    vm32 = awkward.forth.ForthMachine32(": foo if if 999 then then ; -1 0 foo")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32(": foo if if 999 then then ; 0 0 foo")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32(": foo if if 999 then then ; 1 0 foo")
    vm32.run()
    assert vm32.stack == [1]

    vm32 = awkward.forth.ForthMachine32(": foo if if 999 then then ; -1 1 foo")
    vm32.run()
    assert vm32.stack == [999]

    vm32 = awkward.forth.ForthMachine32(": foo if if 999 then then ; 0 1 foo")
    vm32.run()
    assert vm32.stack == []

    vm32 = awkward.forth.ForthMachine32(": foo if if 999 then then ; 1 1 foo")
    vm32.run()
    assert vm32.stack == [999]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else 123 then ; -1 foo")
    vm32.run()
    assert vm32.stack == [999]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else 123 then ; 0 foo")
    vm32.run()
    assert vm32.stack == [123]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else 123 then ; 1 foo")
    vm32.run()
    assert vm32.stack == [999]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else 123 then ; -1 -1 foo")
    vm32.run()
    assert vm32.stack == [-1, 999]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else 123 then ; 0 -1 foo")
    vm32.run()
    assert vm32.stack == [0, 999]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else 123 then ; 1 -1 foo")
    vm32.run()
    assert vm32.stack == [1, 999]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else 123 then ; -1 0 foo")
    vm32.run()
    assert vm32.stack == [-1, 123]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else 123 then ; 0 0 foo")
    vm32.run()
    assert vm32.stack == [0, 123]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else 123 then ; 1 0 foo")
    vm32.run()
    assert vm32.stack == [1, 123]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else 123 then ; -1 1 foo")
    vm32.run()
    assert vm32.stack == [-1, 999]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else 123 then ; 0 1 foo")
    vm32.run()
    assert vm32.stack == [0, 999]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else 123 then ; 1 1 foo")
    vm32.run()
    assert vm32.stack == [1, 999]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if if 999 then else 123 then ; -1 -1 foo"
    )
    vm32.run()
    assert vm32.stack == [999]

    vm32 = awkward.forth.ForthMachine32(": foo if if 999 then else 123 then ; 0 -1 foo")
    vm32.run()
    assert vm32.stack == []

    vm32 = awkward.forth.ForthMachine32(": foo if if 999 then else 123 then ; 1 -1 foo")
    vm32.run()
    assert vm32.stack == [999]

    vm32 = awkward.forth.ForthMachine32(": foo if if 999 then else 123 then ; -1 0 foo")
    vm32.run()
    assert vm32.stack == [-1, 123]

    vm32 = awkward.forth.ForthMachine32(": foo if if 999 then else 123 then ; 0 0 foo")
    vm32.run()
    assert vm32.stack == [0, 123]

    vm32 = awkward.forth.ForthMachine32(": foo if if 999 then else 123 then ; 1 0 foo")
    vm32.run()
    assert vm32.stack == [1, 123]

    vm32 = awkward.forth.ForthMachine32(": foo if if 999 then else 123 then ; -1 1 foo")
    vm32.run()
    assert vm32.stack == [999]

    vm32 = awkward.forth.ForthMachine32(": foo if if 999 then else 123 then ; 0 1 foo")
    vm32.run()
    assert vm32.stack == []

    vm32 = awkward.forth.ForthMachine32(": foo if if 999 then else 123 then ; 1 1 foo")
    vm32.run()
    assert vm32.stack == [999]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if 999 else if 123 then then ; -1 -1 foo"
    )
    vm32.run()
    assert vm32.stack == [-1, 999]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else if 123 then then ; 0 -1 foo")
    vm32.run()
    assert vm32.stack == [0, 999]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else if 123 then then ; 1 -1 foo")
    vm32.run()
    assert vm32.stack == [1, 999]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else if 123 then then ; -1 0 foo")
    vm32.run()
    assert vm32.stack == [123]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else if 123 then then ; 0 0 foo")
    vm32.run()
    assert vm32.stack == []

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else if 123 then then ; 1 0 foo")
    vm32.run()
    assert vm32.stack == [123]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else if 123 then then ; -1 1 foo")
    vm32.run()
    assert vm32.stack == [-1, 999]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else if 123 then then ; 0 1 foo")
    vm32.run()
    assert vm32.stack == [0, 999]

    vm32 = awkward.forth.ForthMachine32(": foo if 999 else if 123 then then ; 1 1 foo")
    vm32.run()
    assert vm32.stack == [1, 999]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if if 999 else 321 then else 123 then ; -1 -1 foo"
    )
    vm32.run()
    assert vm32.stack == [999]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if if 999 else 321 then else 123 then ; 0 -1 foo"
    )
    vm32.run()
    assert vm32.stack == [321]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if if 999 else 321 then else 123 then ; 1 -1 foo"
    )
    vm32.run()
    assert vm32.stack == [999]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if if 999 else 321 then else 123 then ; -1 0 foo"
    )
    vm32.run()
    assert vm32.stack == [-1, 123]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if if 999 else 321 then else 123 then ; 0 0 foo"
    )
    vm32.run()
    assert vm32.stack == [0, 123]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if if 999 else 321 then else 123 then ; 1 0 foo"
    )
    vm32.run()
    assert vm32.stack == [1, 123]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if if 999 else 321 then else 123 then ; -1 1 foo"
    )
    vm32.run()
    assert vm32.stack == [999]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if if 999 else 321 then else 123 then ; 0 1 foo"
    )
    vm32.run()
    assert vm32.stack == [321]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if if 999 else 321 then else 123 then ; 1 1 foo"
    )
    vm32.run()
    assert vm32.stack == [999]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if 999 else if 123 else 321 then then ; -1 -1 foo"
    )
    vm32.run()
    assert vm32.stack == [-1, 999]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if 999 else if 123 else 321 then then ; 0 -1 foo"
    )
    vm32.run()
    assert vm32.stack == [0, 999]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if 999 else if 123 else 321 then then ; 1 -1 foo"
    )
    vm32.run()
    assert vm32.stack == [1, 999]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if 999 else if 123 else 321 then then ; -1 0 foo"
    )
    vm32.run()
    assert vm32.stack == [123]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if 999 else if 123 else 321 then then ; 0 0 foo"
    )
    vm32.run()
    assert vm32.stack == [321]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if 999 else if 123 else 321 then then ; 1 0 foo"
    )
    vm32.run()
    assert vm32.stack == [123]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if 999 else if 123 else 321 then then ; -1 1 foo"
    )
    vm32.run()
    assert vm32.stack == [-1, 999]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if 999 else if 123 else 321 then then ; 0 1 foo"
    )
    vm32.run()
    assert vm32.stack == [0, 999]

    vm32 = awkward.forth.ForthMachine32(
        ": foo if 999 else if 123 else 321 then then ; 1 1 foo"
    )
    vm32.run()
    assert vm32.stack == [1, 999]

    vm32 = awkward.forth.ForthMachine32(": foo do i loop ; 10 5 foo")
    vm32.run()
    assert vm32.stack == [5, 6, 7, 8, 9]

    vm32 = awkward.forth.ForthMachine32(": foo do i i +loop ; 100 5 foo")
    vm32.run()
    assert vm32.stack == [5, 10, 20, 40, 80]

    vm32 = awkward.forth.ForthMachine32(": foo 10 5 do 3 0 do 1+ loop loop ; 1 foo")
    vm32.run()
    assert vm32.stack == [16]

    vm32 = awkward.forth.ForthMachine32(": foo 10 5 do 3 0 do i loop loop ; foo")
    vm32.run()
    assert vm32.stack == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

    vm32 = awkward.forth.ForthMachine32(": foo 10 5 do 3 0 do j loop loop ; foo")
    vm32.run()
    assert vm32.stack == [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]

    vm32 = awkward.forth.ForthMachine32(": foo 10 5 do 3 0 do i j * loop loop ; foo")
    vm32.run()
    assert vm32.stack == [0, 5, 10, 0, 6, 12, 0, 7, 14, 0, 8, 16, 0, 9, 18]

    vm32 = awkward.forth.ForthMachine32(
        ": foo 10 5 do 8 6 do 3 0 do i j * k * loop loop loop ; foo"
    )
    vm32.run()
    assert vm32.stack == [
        0,
        30,
        60,
        0,
        35,
        70,
        0,
        36,
        72,
        0,
        42,
        84,
        0,
        42,
        84,
        0,
        49,
        98,
        0,
        48,
        96,
        0,
        56,
        112,
        0,
        54,
        108,
        0,
        63,
        126,
    ]

    vm32 = awkward.forth.ForthMachine32(": foo 3 begin dup 1 - dup 0= until ; foo")
    vm32.run()
    assert vm32.stack == [3, 2, 1, 0]

    vm32 = awkward.forth.ForthMachine32(
        ": foo 4 begin dup 1 - dup 0= invert while 123 drop repeat ; foo"
    )
    vm32.run()
    assert vm32.stack == [4, 3, 2, 1, 0]

    vm32 = awkward.forth.ForthMachine32(
        ": foo 3 begin dup 1 - dup 0= if exit then again ; foo"
    )
    vm32.run()
    assert vm32.stack == [3, 2, 1, 0]

    vm32 = awkward.forth.ForthMachine32("1 2 3 4 dup")
    vm32.run()
    assert vm32.stack == [1, 2, 3, 4, 4]

    vm32 = awkward.forth.ForthMachine32("1 2 3 4 drop")
    vm32.run()
    assert vm32.stack == [1, 2, 3]

    vm32 = awkward.forth.ForthMachine32("1 2 3 4 swap")
    vm32.run()
    assert vm32.stack == [1, 2, 4, 3]

    vm32 = awkward.forth.ForthMachine32("1 2 3 4 over")
    vm32.run()
    assert vm32.stack == [1, 2, 3, 4, 3]

    vm32 = awkward.forth.ForthMachine32("1 2 3 4 rot")
    vm32.run()
    assert vm32.stack == [1, 3, 4, 2]

    vm32 = awkward.forth.ForthMachine32("1 2 3 4 nip")
    vm32.run()
    assert vm32.stack == [1, 2, 4]

    vm32 = awkward.forth.ForthMachine32("1 2 3 4 tuck")
    vm32.run()
    assert vm32.stack == [1, 2, 4, 3, 4]

    vm32 = awkward.forth.ForthMachine32("3 5 +")
    vm32.run()
    assert vm32.stack == [8]

    vm32 = awkward.forth.ForthMachine32("-3 5 +")
    vm32.run()
    assert vm32.stack == [2]

    vm32 = awkward.forth.ForthMachine32("3 -5 +")
    vm32.run()
    assert vm32.stack == [-2]

    vm32 = awkward.forth.ForthMachine32("-3 -5 +")
    vm32.run()
    assert vm32.stack == [-8]

    vm32 = awkward.forth.ForthMachine32("3 5 -")
    vm32.run()
    assert vm32.stack == [-2]

    vm32 = awkward.forth.ForthMachine32("-3 5 -")
    vm32.run()
    assert vm32.stack == [-8]

    vm32 = awkward.forth.ForthMachine32("3 -5 -")
    vm32.run()
    assert vm32.stack == [8]

    vm32 = awkward.forth.ForthMachine32("-3 -5 -")
    vm32.run()
    assert vm32.stack == [2]

    vm32 = awkward.forth.ForthMachine32("5 3 -")
    vm32.run()
    assert vm32.stack == [2]

    vm32 = awkward.forth.ForthMachine32("5 -3 -")
    vm32.run()
    assert vm32.stack == [8]

    vm32 = awkward.forth.ForthMachine32("-5 3 -")
    vm32.run()
    assert vm32.stack == [-8]

    vm32 = awkward.forth.ForthMachine32("-5 -3 -")
    vm32.run()
    assert vm32.stack == [-2]

    vm32 = awkward.forth.ForthMachine32("3 5 *")
    vm32.run()
    assert vm32.stack == [15]

    vm32 = awkward.forth.ForthMachine32("-3 5 *")
    vm32.run()
    assert vm32.stack == [-15]

    vm32 = awkward.forth.ForthMachine32("3 -5 *")
    vm32.run()
    assert vm32.stack == [-15]

    vm32 = awkward.forth.ForthMachine32("-3 -5 *")
    vm32.run()
    assert vm32.stack == [15]

    vm32 = awkward.forth.ForthMachine32("22 7 /")
    vm32.run()
    assert vm32.stack == [3]

    vm32 = awkward.forth.ForthMachine32("-22 7 /")
    vm32.run()
    assert vm32.stack == [-4]

    vm32 = awkward.forth.ForthMachine32("22 -7 /")
    vm32.run()
    assert vm32.stack == [-4]

    vm32 = awkward.forth.ForthMachine32("-22 -7 /")
    vm32.run()
    assert vm32.stack == [3]

    vm32 = awkward.forth.ForthMachine32("22 7 mod")
    vm32.run()
    assert vm32.stack == [1]

    vm32 = awkward.forth.ForthMachine32("-22 7 mod")
    vm32.run()
    assert vm32.stack == [6]

    vm32 = awkward.forth.ForthMachine32("22 -7 mod")
    vm32.run()
    assert vm32.stack == [-6]

    vm32 = awkward.forth.ForthMachine32("-22 -7 mod")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("22 7 /mod")
    vm32.run()
    assert vm32.stack == [1, 3]

    vm32 = awkward.forth.ForthMachine32("-22 7 /mod")
    vm32.run()
    assert vm32.stack == [6, -4]

    vm32 = awkward.forth.ForthMachine32("22 -7 /mod")
    vm32.run()
    assert vm32.stack == [-6, -4]

    vm32 = awkward.forth.ForthMachine32("-22 -7 /mod")
    vm32.run()
    assert vm32.stack == [-1, 3]

    vm32 = awkward.forth.ForthMachine32("-2 abs")
    vm32.run()
    assert vm32.stack == [2]

    vm32 = awkward.forth.ForthMachine32("-1 abs")
    vm32.run()
    assert vm32.stack == [1]

    vm32 = awkward.forth.ForthMachine32("0 abs")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("1 abs")
    vm32.run()
    assert vm32.stack == [1]

    vm32 = awkward.forth.ForthMachine32("2 abs")
    vm32.run()
    assert vm32.stack == [2]

    vm32 = awkward.forth.ForthMachine32("5 5 min")
    vm32.run()
    assert vm32.stack == [5]

    vm32 = awkward.forth.ForthMachine32("3 -5 min")
    vm32.run()
    assert vm32.stack == [-5]

    vm32 = awkward.forth.ForthMachine32("-3 5 min")
    vm32.run()
    assert vm32.stack == [-3]

    vm32 = awkward.forth.ForthMachine32("3 5 min")
    vm32.run()
    assert vm32.stack == [3]

    vm32 = awkward.forth.ForthMachine32("5 5 max")
    vm32.run()
    assert vm32.stack == [5]

    vm32 = awkward.forth.ForthMachine32("3 -5 max")
    vm32.run()
    assert vm32.stack == [3]

    vm32 = awkward.forth.ForthMachine32("-3 5 max")
    vm32.run()
    assert vm32.stack == [5]

    vm32 = awkward.forth.ForthMachine32("3 5 max")
    vm32.run()
    assert vm32.stack == [5]

    vm32 = awkward.forth.ForthMachine32("-2 negate")
    vm32.run()
    assert vm32.stack == [2]

    vm32 = awkward.forth.ForthMachine32("-1 negate")
    vm32.run()
    assert vm32.stack == [1]

    vm32 = awkward.forth.ForthMachine32("0 negate")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("1 negate")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("2 negate")
    vm32.run()
    assert vm32.stack == [-2]

    vm32 = awkward.forth.ForthMachine32("-1 1+")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("0 1+")
    vm32.run()
    assert vm32.stack == [1]

    vm32 = awkward.forth.ForthMachine32("1 1+")
    vm32.run()
    assert vm32.stack == [2]

    vm32 = awkward.forth.ForthMachine32("-1 1-")
    vm32.run()
    assert vm32.stack == [-2]

    vm32 = awkward.forth.ForthMachine32("0 1-")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("1 1-")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("-1 0=")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("0 0=")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("1 0=")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("5 5 =")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("3 -5 =")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("-3 5 =")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("3 5 =")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("5 5 <>")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("3 -5 <>")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("-3 5 <>")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("3 5 <>")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("5 5 >")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("3 -5 >")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("-3 5 >")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("3 5 >")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("5 5 >=")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("3 -5 >=")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("-3 5 >=")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("3 5 >=")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("5 5 <")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("3 -5 <")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("-3 5 <")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("3 5 <")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("5 5 <=")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("3 -5 <=")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("-3 5 <=")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("3 5 <=")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("-1 -1 and")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("0 -1 and")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("1 -1 and")
    vm32.run()
    assert vm32.stack == [1]

    vm32 = awkward.forth.ForthMachine32("-1 0 and")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("0 0 and")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("1 0 and")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("-1 1 and")
    vm32.run()
    assert vm32.stack == [1]

    vm32 = awkward.forth.ForthMachine32("0 1 and")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("1 1 and")
    vm32.run()
    assert vm32.stack == [1]

    vm32 = awkward.forth.ForthMachine32("-1 -1 or")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("0 -1 or")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("1 -1 or")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("-1 0 or")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("0 0 or")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("1 0 or")
    vm32.run()
    assert vm32.stack == [1]

    vm32 = awkward.forth.ForthMachine32("-1 1 or")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("0 1 or")
    vm32.run()
    assert vm32.stack == [1]

    vm32 = awkward.forth.ForthMachine32("1 1 or")
    vm32.run()
    assert vm32.stack == [1]

    vm32 = awkward.forth.ForthMachine32("-1 -1 xor")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("0 -1 xor")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("1 -1 xor")
    vm32.run()
    assert vm32.stack == [-2]

    vm32 = awkward.forth.ForthMachine32("-1 0 xor")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("0 0 xor")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("1 0 xor")
    vm32.run()
    assert vm32.stack == [1]

    vm32 = awkward.forth.ForthMachine32("-1 1 xor")
    vm32.run()
    assert vm32.stack == [-2]

    vm32 = awkward.forth.ForthMachine32("0 1 xor")
    vm32.run()
    assert vm32.stack == [1]

    vm32 = awkward.forth.ForthMachine32("1 1 xor")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("-1 invert")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("0 invert")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("1 invert")
    vm32.run()
    assert vm32.stack == [-2]

    vm32 = awkward.forth.ForthMachine32("true")
    vm32.run()
    assert vm32.stack == [-1]

    vm32 = awkward.forth.ForthMachine32("false")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32(": foo 3 + 2 * ; 4 foo foo")
    vm32.run()
    assert vm32.stack == [34]

    vm32 = awkward.forth.ForthMachine32(": foo 3 + 2 * ; : bar 4 foo foo ; bar")
    vm32.run()
    assert vm32.stack == [34]

    vm32 = awkward.forth.ForthMachine32(
        ": factorial dup 2 < if drop 1 exit then dup 1- recurse * ; 5 factorial"
    )
    vm32.run()
    assert vm32.stack == [120]

    vm32 = awkward.forth.ForthMachine32("variable x 10 x ! 5 x +! x @ x @ x @")
    vm32.run()
    assert vm32.stack == [15, 15, 15]

    vm32 = awkward.forth.ForthMachine32(
        "variable x 10 x ! 5 x +! : foo x @ x @ x @ ; foo foo"
    )
    vm32.run()
    assert vm32.stack == [15, 15, 15, 15, 15, 15]


def test_gforth_lshift_rshift():
    vm32 = awkward.forth.ForthMachine32("0 1 lshift")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("0 2 lshift")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("0 3 lshift")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("1 1 lshift")
    vm32.run()
    assert vm32.stack == [2]

    vm32 = awkward.forth.ForthMachine32("1 2 lshift")
    vm32.run()
    assert vm32.stack == [4]

    vm32 = awkward.forth.ForthMachine32("1 3 lshift")
    vm32.run()
    assert vm32.stack == [8]

    vm32 = awkward.forth.ForthMachine32("5 1 lshift")
    vm32.run()
    assert vm32.stack == [10]

    vm32 = awkward.forth.ForthMachine32("5 2 lshift")
    vm32.run()
    assert vm32.stack == [20]

    vm32 = awkward.forth.ForthMachine32("5 3 lshift")
    vm32.run()
    assert vm32.stack == [40]

    vm32 = awkward.forth.ForthMachine32("-5 1 lshift")
    vm32.run()
    assert vm32.stack == [-10]

    vm32 = awkward.forth.ForthMachine32("-5 2 lshift")
    vm32.run()
    assert vm32.stack == [-20]

    vm32 = awkward.forth.ForthMachine32("-5 3 lshift")
    vm32.run()
    assert vm32.stack == [-40]

    vm32 = awkward.forth.ForthMachine32("0 1 rshift")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("0 2 rshift")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("0 3 rshift")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("1 1 rshift")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("1 2 rshift")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("1 3 rshift")
    vm32.run()
    assert vm32.stack == [0]

    vm32 = awkward.forth.ForthMachine32("5 1 rshift")
    vm32.run()
    assert vm32.stack == [2]

    vm32 = awkward.forth.ForthMachine32("5 2 rshift")
    vm32.run()
    assert vm32.stack == [1]

    vm32 = awkward.forth.ForthMachine32("5 3 rshift")
    vm32.run()
    assert vm32.stack == [0]

    # I don't understand what gforth's rshift is doing with negative numbers.
    # These can't be brought into agreement even with 64-bit machines.
    # Maybe gforth is just wrong: I would expect the results WE get, not THEIRS.

    vm32 = awkward.forth.ForthMachine32("-5 1 rshift")
    vm32.run()
    # assert vm32.stack == [9223372036854775805]
    assert vm32.stack == [-3]

    vm32 = awkward.forth.ForthMachine32("-5 2 rshift")
    vm32.run()
    # assert vm32.stack == [4611686018427387902]
    assert vm32.stack == [-2]

    vm32 = awkward.forth.ForthMachine32("-5 3 rshift")
    vm32.run()
    # assert vm32.stack == [2305843009213693951]
    assert vm32.stack == [-1]


def test_input_output():
    vm32 = awkward.forth.ForthMachine32(
        """
output y float32
4 3 2 1
y <- stack
y <- stack
y <- stack
y <- stack
"""
    )
    vm32.run()
    assert vm32.stack == []
    assert ak.to_list(vm32["y"]) == [1, 2, 3, 4]

    vm32 = awkward.forth.ForthMachine32(
        """
output y float32
4 3 2 1
y <- stack
y <- stack
1 y rewind
y <- stack
y <- stack
"""
    )
    vm32.run()
    assert ak.to_list(vm32["y"]) == [1, 3, 4]

    vm32 = awkward.forth.ForthMachine32(
        """
input x
x i-> stack
x i-> stack
-8 x skip
x i-> stack
x i-> stack
x i-> stack
x i-> stack
"""
    )
    vm32.run({"x": np.array([1, 2, 3, 4], np.int32)})
    assert vm32.stack == [1, 2, 1, 2, 3, 4]

    vm32 = awkward.forth.ForthMachine32(
        """
input x
4 x #i-> stack
"""
    )
    vm32.run({"x": np.array([1, 2, 3, 4], np.int32)})
    assert vm32.stack == [1, 2, 3, 4]

    vm32 = awkward.forth.ForthMachine32(
        """
input x
4 x #!i-> stack
"""
    )
    vm32.run({"x": np.array([1, 2, 3, 4], np.int32)})
    assert vm32.stack == [16777216, 33554432, 50331648, 67108864]

    vm32 = awkward.forth.ForthMachine32(
        """
input x
8 x #h-> stack
"""
    )
    vm32.run({"x": np.array([1, 2, 3, 4], np.int32)})
    assert vm32.stack == [1, 0, 2, 0, 3, 0, 4, 0]

    vm32 = awkward.forth.ForthMachine32(
        """
input x output y float32
x i-> y
x i-> y
x i-> y
x i-> y
"""
    )
    vm32.run({"x": np.array([1, 2, 3, 4], np.int32)})
    assert vm32.stack == []
    assert ak.to_list(vm32["y"]) == [1, 2, 3, 4]

    vm32 = awkward.forth.ForthMachine32(
        """
input x output y float32
4 x #i-> y
"""
    )
    vm32.run({"x": np.array([1, 2, 3, 4], np.int32)})
    assert vm32.stack == []
    assert ak.to_list(vm32["y"]) == [1, 2, 3, 4]


def test_uproot():
    uproot = pytest.importorskip("uproot")
    skhep_testdata = pytest.importorskip("skhep_testdata")

    vector_of_vectors = uproot.open(
        skhep_testdata.data_path("uproot-vectorVectorDouble.root")
    )["t/x"]
    inputs = {
        "data": np.array(vector_of_vectors.basket(0).data),
        "byte_offsets": vector_of_vectors.basket(0).byte_offsets,
    }

    vm32 = awkward.forth.ForthMachine32(
        """
input data
input byte_offsets
output outer-offsets int32
output inner-offsets int32
output content-data float64
variable length
variable outer-last
variable inner-last

0 length !
0 outer-last !
0 inner-last !
outer-last @ outer-offsets <- stack
inner-last @ inner-offsets <- stack

begin
  byte_offsets i-> stack
  6 + data seek
  1 length +!

  data !i-> stack dup outer-last +!
  outer-last @ outer-offsets <- stack

  0 do
    data !i-> stack dup inner-last +!
    inner-last @ inner-offsets <- stack

    data #!d-> content-data
  loop
again
"""
    )
    vm32.run(inputs, raise_read_beyond=False, raise_seek_beyond=False)
    assert ak.to_list(vm32["outer-offsets"]) == [0, 0, 2, 5, 6, 9]
    assert ak.to_list(vm32["inner-offsets"]) == [0, 0, 0, 1, 1, 3, 6, 7, 8, 9]
    assert ak.to_list(vm32["content-data"]) == [
        10,
        10,
        20,
        20,
        -21,
        -22,
        200,
        -201,
        202,
    ]


def test_varint():
    # https://lucene.apache.org/core/3_5_0/fileformats.html#VInt
    data = np.array(
        [0, 1, 2, 127, 128, 1, 129, 1, 130, 1, 255, 127, 128, 128, 1, 129, 128, 1],
        np.uint8,
    )

    vm32 = awkward.forth.ForthMachine32(
        """
input x
10 0 do
  x varint-> stack
loop
"""
    )
    vm32.run({"x": data})
    assert vm32.stack == [0, 1, 2, 127, 128, 129, 130, 16383, 16384, 16385]

    vm32 = awkward.forth.ForthMachine32(
        """
input x
10 x #varint-> stack
"""
    )
    vm32.run({"x": data})
    assert vm32.stack == [0, 1, 2, 127, 128, 129, 130, 16383, 16384, 16385]

    vm32 = awkward.forth.ForthMachine32(
        """
input x
output y int32
10 0 do
  x varint-> y
loop
"""
    )
    vm32.run({"x": data})
    assert ak.to_list(vm32["y"]) == [0, 1, 2, 127, 128, 129, 130, 16383, 16384, 16385]

    vm32 = awkward.forth.ForthMachine32(
        """
input x
output y int32
10 x #varint-> y
"""
    )
    vm32.run({"x": data})
    assert ak.to_list(vm32["y"]) == [0, 1, 2, 127, 128, 129, 130, 16383, 16384, 16385]


def test_zigzag():
    # https://developers.google.com/protocol-buffers/docs/encoding?csw=1#types
    data = np.array(
        [0, 1, 2, 3, 254, 255, 255, 255, 15, 255, 255, 255, 255, 15], np.uint8
    )

    vm32 = awkward.forth.ForthMachine32(
        """
input x
6 0 do
  x zigzag-> stack
loop
"""
    )
    vm32.run({"x": data})
    assert vm32.stack == [0, -1, 1, -2, 2147483647, -2147483648]

    vm32 = awkward.forth.ForthMachine32(
        """
input x
6 x #zigzag-> stack
"""
    )
    vm32.run({"x": data})
    assert vm32.stack == [0, -1, 1, -2, 2147483647, -2147483648]

    vm32 = awkward.forth.ForthMachine32(
        """
input x
output y int32
6 0 do
  x zigzag-> y
loop
"""
    )
    vm32.run({"x": data})
    assert ak.to_list(vm32["y"]) == [0, -1, 1, -2, 2147483647, -2147483648]

    vm32 = awkward.forth.ForthMachine32(
        """
input x
output y int32
6 x #zigzag-> y
"""
    )
    vm32.run({"x": data})
    assert ak.to_list(vm32["y"]) == [0, -1, 1, -2, 2147483647, -2147483648]


def test_nbit_little():
    data = np.array(
        [252, 255, 191, 255, 251, 255, 255, 239, 251, 255, 191, 255], np.uint8
    )
    expectation = (
        [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]
        + [3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3]
        + [3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3]
        + [3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3]
    )

    vm32 = awkward.forth.ForthMachine32("input x 48 x #2bit-> stack")
    vm32.run({"x": data})
    assert vm32.stack == expectation
    assert vm32.input_position("x") == 12

    vm32 = awkward.forth.ForthMachine32("input x 47 x #2bit-> stack")
    vm32.run({"x": data})
    assert vm32.stack == expectation[:47]
    assert vm32.input_position("x") == 12

    vm32 = awkward.forth.ForthMachine32("input x 45 x #2bit-> stack")
    vm32.run({"x": data})
    assert vm32.stack == expectation[:45]
    assert vm32.input_position("x") == 12

    vm32 = awkward.forth.ForthMachine32("input x 44 x #2bit-> stack")
    vm32.run({"x": data})
    assert vm32.stack == expectation[:44]
    assert vm32.input_position("x") == 11

    vm32 = awkward.forth.ForthMachine32("input x 43 x #2bit-> stack")
    vm32.run({"x": data})
    assert vm32.stack == expectation[:43]
    assert vm32.input_position("x") == 11

    vm32 = awkward.forth.ForthMachine32("input x 1 x #2bit-> stack")
    vm32.run({"x": data})
    assert vm32.stack == expectation[:1]
    assert vm32.input_position("x") == 1

    vm32 = awkward.forth.ForthMachine32("input x x 2bit-> stack")
    vm32.run({"x": data})
    assert vm32.stack == expectation[:1]
    assert vm32.input_position("x") == 1

    vm32 = awkward.forth.ForthMachine32("input x 0 x #2bit-> stack")
    vm32.run({"x": data})
    assert vm32.stack == []
    assert vm32.input_position("x") == 0

    vm32 = awkward.forth.ForthMachine32("input x output y int32 48 x #2bit-> y")
    vm32.run({"x": data})
    assert ak.to_list(vm32["y"]) == expectation
    assert vm32.input_position("x") == 12

    vm32 = awkward.forth.ForthMachine32("input x output y int32 x 2bit-> y")
    vm32.run({"x": data})
    assert ak.to_list(vm32["y"]) == expectation[:1]
    assert vm32.input_position("x") == 1


def test_nbit_big():
    data = np.array(
        [63, 255, 253, 255, 223, 255, 255, 247, 223, 255, 253, 255], np.uint8
    )
    expectation = (
        [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]
        + [3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3]
        + [3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3]
        + [3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3]
    )

    vm32 = awkward.forth.ForthMachine32("input x 48 x #!2bit-> stack")
    vm32.run({"x": data})
    assert vm32.stack == expectation
    assert vm32.input_position("x") == 12

    vm32 = awkward.forth.ForthMachine32("input x 47 x #!2bit-> stack")
    vm32.run({"x": data})
    assert vm32.stack == expectation[:47]
    assert vm32.input_position("x") == 12

    vm32 = awkward.forth.ForthMachine32("input x 45 x #!2bit-> stack")
    vm32.run({"x": data})
    assert vm32.stack == expectation[:45]
    assert vm32.input_position("x") == 12

    vm32 = awkward.forth.ForthMachine32("input x 44 x #!2bit-> stack")
    vm32.run({"x": data})
    assert vm32.stack == expectation[:44]
    assert vm32.input_position("x") == 11

    vm32 = awkward.forth.ForthMachine32("input x 43 x #!2bit-> stack")
    vm32.run({"x": data})
    assert vm32.stack == expectation[:43]
    assert vm32.input_position("x") == 11

    vm32 = awkward.forth.ForthMachine32("input x 1 x #!2bit-> stack")
    vm32.run({"x": data})
    assert vm32.stack == expectation[:1]
    assert vm32.input_position("x") == 1

    vm32 = awkward.forth.ForthMachine32("input x x !2bit-> stack")
    vm32.run({"x": data})
    assert vm32.stack == expectation[:1]
    assert vm32.input_position("x") == 1

    vm32 = awkward.forth.ForthMachine32("input x 0 x #!2bit-> stack")
    vm32.run({"x": data})
    assert vm32.stack == []
    assert vm32.input_position("x") == 0

    vm32 = awkward.forth.ForthMachine32("input x output y int32 48 x #!2bit-> y")
    vm32.run({"x": data})
    assert ak.to_list(vm32["y"]) == expectation
    assert vm32.input_position("x") == 12

    vm32 = awkward.forth.ForthMachine32("input x output y int32 x !2bit-> y")
    vm32.run({"x": data})
    assert ak.to_list(vm32["y"]) == expectation[:1]
    assert vm32.input_position("x") == 1


def test_output_dup():
    vm = awkward.forth.ForthMachine32(
        """
output stuff int32
1 2 3 4
stuff <- stack
10 stuff dup
"""
    )
    vm.run()
    assert ak.to_list(vm["stuff"]) == [4] * 11
    assert vm.stack == [1, 2, 3]


def test_decompile_complex():
    read_repetition_levels = awkward.forth.ForthMachine32(
        """
input stream
output replevels uint8

stream I-> stack
begin
  stream varint-> stack
  dup 1 and 0= if
    ( run-length encoding )
    stream {rle_format}-> replevels
    1 rshift 1-
    replevels dup
  else
    ( bit-packed )
    1 rshift 8 *
    stream #{bit_width}bit-> replevels
  then
  dup stream pos 4 - <=
until
""".format(
            bit_width=2, rle_byte_width=1, rle_format="B"
        )
    )
    assert (
        read_repetition_levels.decompiled
        == """input stream
output replevels uint8

stream I-> stack
begin
  stream varint-> stack
  dup
  1
  and
  0=
  if
    stream B-> replevels
    1
    rshift
    1-
    replevels dup
  else
    1
    rshift
    8
    *
    stream #2bit-> replevels
  then
  dup
  stream pos
  4
  -
  <=
until
"""
    )


def test_parse_strings():
    vm32 = awkward.forth.ForthMachine32(
        """
." hello there      long space"
"""
    )
    assert (
        vm32.decompiled
        == """." hello there      long space"
"""
    )
