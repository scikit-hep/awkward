# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test():
    array = awkward1.Array([[]])
    # print(awkward1.choose(array, 3, diagonal=False))
    # raise Exception
