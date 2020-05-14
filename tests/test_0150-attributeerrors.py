# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

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

    recordarray = awkward1.from_iter([{"x": 1}, {"x": 2}, {"x": 3}], highlevel=False)
    recordarray.setparameter("__record__", "Dummy")
    array = awkward1.Array(recordarray, behavior=behavior)

    with pytest.raises(AttributeError) as err:
        array[1].broken
    assert str(err.value) == "I'm broken!"   # not "no field named 'broken'"
