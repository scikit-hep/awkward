# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test():
    array = awkward1.Array([{"x": "one"}, {"x": "two"}, {"x": "three"}])
    assert awkward1.tolist(array) == [{"x": "one"}, {"x": "two"}, {"x": "three"}]
    assert awkward1.tolist(awkward1.fromiter(awkward1.tolist(array))) == [{"x": "one"}, {"x": "two"}, {"x": "three"}]
    assert awkward1.tolist(array.layout) == [{"x": "one"}, {"x": "two"}, {"x": "three"}]
    assert awkward1.tolist(awkward1.fromiter(awkward1.tolist(array.layout))) == [{"x": "one"}, {"x": "two"}, {"x": "three"}]
