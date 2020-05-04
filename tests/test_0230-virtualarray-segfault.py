# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1
import awkward1._ext

def test_0230():
    rec = awkward1.zip({"x": awkward1.virtual(lambda: awkward1.Array([1, 2, 3, 4]), length=4),},
                       depth_limit=1)
    assert awkward1.to_list(rec.x[1:]) == [2, 3, 4]
    assert awkward1.to_list(rec.x[1:]*2) == [4, 6, 8]
