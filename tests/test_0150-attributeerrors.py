# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

class Dummy(awkward1.Record):
    @property
    def broken(self):
        raise AttributeError("I'm broken!")

def test():
    behavior = {}
    behavior["Dummy"] = Dummy

    array = awkward1.Array([{"x": 1}, {"x": 2}, {"x": 3}], behavior=behavior)
    array.layout.setparameter("__record__", "Dummy")

    with pytest.raises(AttributeError) as err:
        array[1].broken
    assert str(err.value) == "I'm broken!"   # not "no field named 'broken'"
