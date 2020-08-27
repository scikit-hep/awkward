# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test():
    np_content = numpy.asfortranarray(numpy.arange(15).reshape(3, 5))
    ak_content = awkward1.layout.NumpyArray(np_content)
    offsets = awkward1.layout.Index64(numpy.array([0, 0, 1, 1, 2, 2, 3, 3]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, ak_content)
    assert awkward1.to_list(listoffsetarray[1, 0]) == [0, 1, 2, 3, 4]
    assert awkward1.to_list(listoffsetarray[3, 0]) == [5, 6, 7, 8, 9]
