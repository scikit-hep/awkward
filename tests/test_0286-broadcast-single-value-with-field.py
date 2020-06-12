# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_broadcast_single_bool():
    base = awkward1.Array([[{"x" : 0.1, "y" : 0.2, "z" : 0.3}, {"x" : 0.4, "y" : 0.5, "z" : 0.6}]])
    base_new1 = awkward1.with_field(base, True, "always_true")
    assert awkward1.to_list(base_new1.always_true) == [[True, True]]
    base_new2 = awkward1.with_field(base_new1, base.x > 0.3, "sometimes_true")
    assert awkward1.to_list(base_new2.always_true) == [[True, True]]
