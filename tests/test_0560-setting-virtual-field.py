# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test():
    array = awkward1.Array([{"x": 1}])
    array["y"] = awkward1.virtual(lambda: [2], cache={}, length=1)

    print("NOW", type(array.caches[0]), id(array.caches[0]))


    awkward1.to_list(array)
