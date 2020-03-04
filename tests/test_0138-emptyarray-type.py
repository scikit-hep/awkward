# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test():
    empty1 = awkward1.Array(awkward1.layout.EmptyArray())
    empty2 = awkward1.Array(awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 0, 0, 0], dtype=numpy.int64)), awkward1.layout.EmptyArray()))
    array = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

    awkward1.tonumpy(empty1).dtype.type is numpy.float64

    awkward1.tolist(array[empty1]) == []
    awkward1.tolist(array[empty1,]) == []
    awkward1.tolist(array[empty2]) == [[], [], []]
    awkward1.tolist(array[empty2,]) == [[], [], []]
