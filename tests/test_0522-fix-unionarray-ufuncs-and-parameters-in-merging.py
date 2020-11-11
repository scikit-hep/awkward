# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test():
    content1 = awkward1.Array([0.0, 1.1, 2.2, 3.3, 4.4]).layout
    content2 = awkward1.Array([  0, 100, 200, 300, 400]).layout
    tags = awkward1.layout.Index8(numpy.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1], numpy.int8))
    index = awkward1.layout.Index64(numpy.array([0, 1, 2, 0, 1, 3, 4, 2, 3, 4], numpy.int64))
    unionarray = awkward1.Array(awkward1.layout.UnionArray8_64(tags, index, [content1, content2]))
    assert unionarray.tolist() == [0.0, 1.1, 2.2, 0, 100, 3.3, 4.4, 200, 300, 400]
    assert (unionarray + 10).tolist() == [10.0, 11.1, 12.2, 10, 110, 13.3, 14.4, 210, 310, 410]
    # assert (10 + unionarray).tolist() == [10.0, 11.1, 12.2, 10, 110, 13.3, 14.4, 210, 310, 410]
