# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward.forth import ForthMachine32


def test_skip_ws():
    vm = ForthMachine32("input x  x skipws")
    vm.run({"x": np.array([ord(x) for x in "     hello"], np.uint8)})
    assert vm.input_position("x") == 5


def test_textint():
    vm = ForthMachine32(
        """input x output y float64
           3 0 do x skipws x textint-> stack loop
           x skipws x textint-> y"""
    )
    vm.run(
        {"x": np.array([ord(x) for x in "     12345 -123      3210  -42"], np.uint8)}
    )
    assert vm.stack == [12345, -123, 3210]
    assert np.asarray(vm["y"]).tolist() == [-42.0]


def test_textfloat():
    vm = ForthMachine32(
        """input x output y float64
           6 0 do x skipws x textfloat-> y loop"""
    )
    vm.run(
        {
            "x": np.array(
                [ord(x) for x in "     42 -42 42.123 -42.123e1 42.123e+2 -42.123e-2"],
                np.uint8,
            )
        }
    )
    assert np.asarray(vm["y"]).tolist() == pytest.approx(
        [42, -42, 42.123, -42.123e1, 42.123e2, -42.123e-2]
    )
