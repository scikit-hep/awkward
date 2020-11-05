# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test_this_should_raise_a_warning():
    awkward1.deprecations_as_errors = True

    one = awkward1.Array([{"x": 1}, {"x": 2}, {"x": 3}])
    two = awkward1.Array([{"x": 1.1}, {"x": 2.2}, {"x": 3.3}])

    with pytest.raises(ValueError):
        one + two

def test_this_should_not():
    def overload_add(left, right):
        return awkward1.Array({"x": left.x + right.x})

    behavior = {}
    behavior[numpy.add, "Overload", "Overload"] = overload_add

    one = awkward1.Array(
        [{"x": 1}, {"x": 2}, {"x": 3}], with_name="Overload", behavior=behavior
    )
    two = awkward1.Array(
        [{"x": 1.1}, {"x": 2.2}, {"x": 3.3}], with_name="Overload", behavior=behavior
    )

    assert (one + two).tolist() == [{"x": 2.1}, {"x": 4.2}, {"x": 6.3}]

