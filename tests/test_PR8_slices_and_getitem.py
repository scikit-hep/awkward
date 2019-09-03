# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_slice():
    print(awkward1.layout.testslice(3))
    # raise Exception
