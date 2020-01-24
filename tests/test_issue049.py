# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

class Point(awkward1.Record):
  def __repr__(self):
    return "<{} {}>".format(self["x"], self["y"])

def test():
    array = awkward1.Array([[{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.0, 0.2]}], [], [{"x": 3, "y": [3.0, 0.3, 3.3]}]])
    array.layout.content.setparameter("__class__", "Point")
    array.layout.content.setparameter("__typestr__", "P")
    awkward1.classes["Point"] = Point
    assert repr(array[0, 0]) == "<1 [1.1]>"
    assert repr(array[0]) == "<Array [<1 [1.1]>, <2 [2, 0.2]>] type='2 * P'>"
    assert repr(array) == "<Array [[<1 [1.1]>, ... <3 [3, 0.3, 3.3]>]] type='3 * var * P'>"
