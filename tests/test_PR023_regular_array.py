# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import os
import json

import pytest
import numpy

import awkward1

content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]));
offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
regulararray = awkward1.layout.RegularArray((2, 3), listoffsetarray)

def test_type():
    assert str(awkward1.typeof(regulararray)) == "2 * 3 * var * float64"

# def test_getitem():
#     print(regulararray[0])
#     print(regulararray[1])
#
#     raise Exception
