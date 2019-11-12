# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import os
import json

import pytest
import numpy
numba = pytest.importorskip("numba")

import awkward1

def test_fromjson():
    a = awkward1.fromjson("[[], [], []]")
    assert awkward1.tolist(a) == [[], [], []]

    # print(a)
    # print(awkward1.tolist(a))
    # print(a.type)
    #
    # raise Exception
