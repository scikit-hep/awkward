# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward.forth


def test_user_defined_error():
    vm32 = awkward.forth.ForthMachine32(
        """
s" this is an error"
"""
    )

    assert (
        vm32.decompiled
        == """s" this is an error"
"""
    )


def test_user_defined_strings():
    vm32 = awkward.forth.ForthMachine32(
        """
s" this is a first string"
s" this is a second string"
s" this is a third string"
s" this is a forth string"
"""
    )

    assert vm32.string_at(0) == "this is a first string"
    assert vm32.string_at(1) == "this is a second string"
    assert vm32.string_at(2) == "this is a third string"
    assert vm32.string_at(3) == "this is a forth string"
    assert vm32.string_at(4) == "a string at 4 is undefined"
    assert vm32.string_at(-1) == "a string at -1 is undefined"


def test_user_defined_exception():
    vm32 = awkward.forth.ForthMachine32(
        """
variable x 0 x !
variable err

s" variable x reached its maximum 10"

: foo
    5 x +! x @
    dup 10 = if
        0 err ! err @ halt
    then
;

0
begin
    foo
    1+
again
"""
    )
    with pytest.raises(ValueError):
        vm32.run()
    assert vm32.stack[-1] == 0
    assert vm32.string_at(vm32.stack[-1]) == "variable x reached its maximum 10"


def test_undefined_error():
    vm32 = awkward.forth.ForthMachine32(
        """
variable x 0 x !

: foo
    5 x +! x @
    dup 10 = if
        halt
    then
;

0
begin
    foo
    1+
again
"""
    )
    with pytest.raises(ValueError) as err:
        vm32.run()
    assert (
        str(err.value)
        == "'user halt' in AwkwardForth runtime: user-defined error or stopping condition"
    )
