# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test():
    array = awkward1.Array([{"x": "one"}, {"x": "two"}, {"x": "three"}], check_valid=True)
    assert awkward1.to_list(array) == [{"x": "one"}, {"x": "two"}, {"x": "three"}]
    assert awkward1.to_list(awkward1.from_iter(awkward1.to_list(array))) == [{"x": "one"}, {"x": "two"}, {"x": "three"}]
    assert awkward1.to_list(array.layout) == [{"x": "one"}, {"x": "two"}, {"x": "three"}]
    assert awkward1.to_list(awkward1.from_iter(awkward1.to_list(array.layout))) == [{"x": "one"}, {"x": "two"}, {"x": "three"}]
