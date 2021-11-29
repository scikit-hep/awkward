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


def test_peek():
    vm = ForthMachine32(
        """input x output y float64
           5 0 do x skipws 0 x peek 1 x peek x textint-> y loop"""
    )
    vm.run(
        {"x": np.array([ord(x) for x in "     12345 -123      3210  -42 98"], np.uint8)}
    )
    assert vm.stack == [
        ord("1"),
        ord("2"),
        ord("-"),
        ord("1"),
        ord("3"),
        ord("2"),
        ord("-"),
        ord("4"),
        ord("9"),
        ord("8"),
    ]
    assert np.asarray(vm["y"]).tolist() == [12345, -123, 3210, -42, 98]


def test_enum():
    vm = ForthMachine32(
        r'input x   3 0 do x skipws x enum s" zero" s" one" s" two" s" three" loop'
    )
    vm.run({"x": np.array([ord(x) for x in "one three two"], np.uint8)})
    assert vm.stack == [1, 3, 2]


def test_general_case():
    vm = ForthMachine32(
        r"case 0 of 1000 endof 1 of 1001 endof 1 1 + of 1002 endof 9999 swap endcase 10 20 30"
    )

    vm.begin()
    vm.stack_push(0)
    vm.resume()
    assert vm.stack == [1000, 10, 20, 30]

    vm.begin()
    vm.stack_push(1)
    vm.resume()
    assert vm.stack == [1001, 10, 20, 30]

    vm.begin()
    vm.stack_push(2)
    vm.resume()
    assert vm.stack == [1002, 10, 20, 30]

    vm.begin()
    vm.stack_push(3)
    vm.resume()
    assert vm.stack == [9999, 10, 20, 30]

    vm.begin()
    vm.stack_push(-1)
    vm.resume()
    assert vm.stack == [9999, 10, 20, 30]

    vm = ForthMachine32(
        r"case 0 of 1000 endof 1 of 1001 endof 1 1 + of 1002 endof endcase 10 20 30"
    )

    vm.begin()
    vm.stack_push(2)
    vm.resume()
    assert vm.stack == [1002, 10, 20, 30]

    vm.begin()
    vm.stack_push(3)
    vm.resume()
    assert vm.stack == [10, 20, 30]

    vm.begin()
    vm.stack_push(-1)
    vm.resume()
    assert vm.stack == [10, 20, 30]


def test_specialized_case():
    vm = ForthMachine32(
        r"case 0 of 1000 endof 1 of 1001 endof 2 of 1002 endof 9999 swap endcase 10 20 30"
    )

    vm.begin()
    vm.stack_push(0)
    vm.resume()
    assert vm.stack == [1000, 10, 20, 30]

    vm.begin()
    vm.stack_push(1)
    vm.resume()
    assert vm.stack == [1001, 10, 20, 30]

    vm.begin()
    vm.stack_push(2)
    vm.resume()
    assert vm.stack == [1002, 10, 20, 30]

    vm.begin()
    vm.stack_push(3)
    vm.resume()
    assert vm.stack == [9999, 10, 20, 30]

    vm.begin()
    vm.stack_push(-1)
    vm.resume()
    assert vm.stack == [9999, 10, 20, 30]

    vm = ForthMachine32(
        r"case 0 of 1000 endof 1 of 1001 endof 2 of 1002 endof endcase 10 20 30"
    )

    vm.begin()
    vm.stack_push(2)
    vm.resume()
    assert vm.stack == [1002, 10, 20, 30]

    vm.begin()
    vm.stack_push(3)
    vm.resume()
    assert vm.stack == [10, 20, 30]

    vm.begin()
    vm.stack_push(-1)
    vm.resume()
    assert vm.stack == [10, 20, 30]

    vm = ForthMachine32(r"case 9999 swap endcase 10 20 30")

    vm.begin()
    vm.stack_push(0)
    vm.resume()
    assert vm.stack == [9999, 10, 20, 30]

    vm = ForthMachine32(r"case endcase 10 20 30")

    vm.begin()
    vm.stack_push(0)
    vm.resume()
    assert vm.stack == [10, 20, 30]

    vm = ForthMachine32(
        r"case 0 of 1000 endof 1 of 1001 endof 2 of endof 9999 swap endcase 10 20 30"
    )

    vm.begin()
    vm.stack_push(1)
    vm.resume()
    assert vm.stack == [1001, 10, 20, 30]

    vm.begin()
    vm.stack_push(2)
    vm.resume()
    assert vm.stack == [10, 20, 30]
