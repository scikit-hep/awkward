# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward.forth import ForthMachine32

pytestmark = pytest.mark.skipif(
    ak._util.py27 or ak._util.py35, reason="JSON doesn't like bytes in old Pythons"
)


def test_skip_ws():
    vm = ForthMachine32("input x  x skipws")
    vm.run({"x": np.array([ord(x) for x in "     hello"], np.uint8)})
    assert vm.input_position("x") == 5


def test_textint():
    vm = ForthMachine32(
        """input x output y float64
           x skipws 2 x #textint-> stack
           x skipws x textint-> stack
           x skipws 2 x #textint-> y"""
    )
    vm.run(
        {"x": np.array([ord(x) for x in "     12345 -123      3210  -42 0"], np.uint8)}
    )
    assert vm.stack == [12345, -123, 3210]
    assert np.asarray(vm["y"]).tolist() == [-42.0, 0.0]


def test_textfloat():
    vm = ForthMachine32(
        """input x output y float64
           x skipws 5 x #textfloat-> y
           x skipws x textfloat-> y"""
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


def test_quotedstr():
    vm = ForthMachine32(
        """input x output y uint8
           x skipws 5 x #quotedstr-> y"""
    )
    vm.run(
        {
            "x": np.array(
                [
                    ord(x)
                    for x in r'     "one" "   two"   "three   " "fo\\u\"r"     "f\niv\u0045"'
                ],
                np.uint8,
            )
        }
    )
    assert (
        ak._v2._util.tobytes(np.asarray(vm["y"])) == b'one   twothree   fo\\u"rf\nivE'
    )
    assert vm.stack == [3, 6, 8, 6, 5]


def test_unicode():
    vm = ForthMachine32(
        """input x output y uint8
           x quotedstr-> y"""
    )
    x = np.array(
        [ord(x) for x in r'"\u0000 \u007f \u0080 \u07ff \u0800 \ud7ff \ue000 \uffff"'],
        np.uint8,
    )
    expecting = json.loads(ak._v2._util.tobytes(x)).encode("utf-8")
    vm.run({"x": x})
    assert ak._v2._util.tobytes(np.asarray(vm["y"])) == expecting
    assert vm.stack == [25] == [len(expecting)]
