# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward.forth


def test_minimal():
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


def test_comments():
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


def test_literal():
    vm32 = awkward.forth.ForthMachine32("1 2 3 4")
    assert ak.to_list(vm32.bytecodes) == [[0, 1, 0, 2, 0, 3, 0, 4]]
    assert vm32.decompiled == """1
2
3
4
"""


def test_def():
    vm32 = awkward.forth.ForthMachine32(": stuff 1 2 3 4 ;")
    assert ak.to_list(vm32.bytecodes) == [[], [0, 1, 0, 2, 0, 3, 0, 4]]
    assert vm32.decompiled == """: stuff
  1
  2
  3
  4
;
"""

    vm32 = awkward.forth.ForthMachine32(": foo 123 : bar 1 2 3 ; 321 ;")
    assert ak.to_list(vm32.bytecodes) == [[], [0, 123, 0, 321], [0, 1, 0, 2, 0, 3]]
    assert vm32.decompiled == """: bar
  1
  2
  3
;

: foo
  123
  321
;
"""

    vm32 = awkward.forth.ForthMachine32(": empty ;")
    assert ak.to_list(vm32.bytecodes) == [[], []]

    print(">" + vm32.decompiled + "<")

    assert vm32.decompiled == """: empty
;
"""

    # vm32 = awkward.forth.ForthMachine32(": infinite recurse ;")
    # print(ak.to_list(vm32.bytecodes))
    # print(vm32.decompiled)
    # raise Exception


# def test_if():
#     vm32 = awkward.forth.ForthMachine32("-1 if 3 5 + then")
#     print(ak.Array(vm32.bytecodes))
#     print(vm32.decompiled)
#     raise Exception
