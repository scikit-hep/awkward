# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test():
    assert awkward1.tolist(awkward1.Record({"x": 1, "y": 2.2})) == {"x": 1, "y": 2.2}
