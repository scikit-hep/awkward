# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_index_slice():
    index = awkward1.layout.Index64(numpy.array(
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        numpy.int64))
    assert index[4] == 400
    assert numpy.asarray(index[3:7]).tolist() == [300, 400, 500, 600]
