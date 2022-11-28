# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


class Pointy(ak.highlevel.Record):
    def __str__(self):
        return "<{} {}>".format(self["x"], self["y"])


def test():
    behavior = {}
    behavior["__typestr__", "Point"] = "P"
    behavior["Point"] = Pointy
    array = ak.highlevel.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.0, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ],
        with_name="Point",
        behavior=behavior,
        check_valid=True,
    )
    assert str(array[0, 0]) == "<1 [1.1]>"
    assert repr(array[0]) == "<Array [<1 [1.1]>, <2 [2, 0.2]>] type='2 * P'>"
    assert (
        repr(array)
        == "<Array [[<1 [1.1]>, <2 [2, 0.2]>], ..., [{...}]] type='3 * var * P'>"
    )
