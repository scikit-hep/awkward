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


# def test_literal():
#     vm32 = awkward.forth.ForthMachine32("1 2 3 4")
#     print(ak.Array(vm32.bytecodes))
#     raise Exception
