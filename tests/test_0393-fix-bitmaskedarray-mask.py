# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test():
    assert numpy.asarray(awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array([1, 1, 1, 1, 1], numpy.int8)), awkward1.layout.NumpyArray(numpy.array([0, 1, 2, 3, 4])), valid_when=True).bytemask()).tolist() == [0, 0, 0, 0, 0]

    assert numpy.asarray(awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array([1, 1, 1, 1, 1], numpy.int8)), awkward1.layout.NumpyArray(numpy.array([0, 1, 2, 3, 4])), valid_when=False).bytemask()).tolist() == [1, 1, 1, 1, 1]

    assert numpy.asarray(awkward1.layout.BitMaskedArray(awkward1.layout.IndexU8(numpy.array([31], numpy.uint8)), awkward1.layout.NumpyArray(numpy.array([0, 1, 2, 3, 4])), valid_when=True, length=5, lsb_order=True).bytemask()).tolist() == [0, 0, 0, 0, 0]

    assert numpy.asarray(awkward1.layout.BitMaskedArray(awkward1.layout.IndexU8(numpy.array([31], numpy.uint8)), awkward1.layout.NumpyArray(numpy.array([0, 1, 2, 3, 4])), valid_when=False, length=5, lsb_order=True).bytemask()).tolist() == [1, 1, 1, 1, 1]
