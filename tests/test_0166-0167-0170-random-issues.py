# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_0166():
    array = awkward1.Array([[2, 3, 5], None, [], [7, 11], None, [13], None, [17, 19]])
    assert awkward1.tolist(awkward1.prod(array, axis=-1)) == [30, None, 1, 77, None, 13, None, 323]

    array = awkward1.Array([[[2, 3], [5]], None, [], [[7], [11]], None, [[13]], None, [[17, 19]]])
    assert awkward1.tolist(awkward1.prod(array, axis=-1)) == [[6, 5], None, [], [7, 11], None, [13], None, [323]]

    array = awkward1.Array([[[2, 3], None, [5]], [], [[7], [11]], [[13]], [None, [17], [19]]])
    awkward1.tolist(awkward1.prod(array, axis=-1)) == [[6, None, 5], [], [7, 11], [13], [None, 17, 19]]

    array = awkward1.Array([[6, None, 5], [], [7, 11], [13], [None, 17, 19]])
    assert awkward1.tolist(awkward1.prod(array, axis=-1)) == [30, 1, 77, 13, 323]

def test_0167():
    pass

def test_0170():
    pass
