# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_mixing_lists_and_none():
    def add(a, b):
        outer = []
        for ai, bi in zip(a, b):
            if ai is None or bi is None:
                outer.append(None)
            else:
                inner = []
                for aj, bj in zip(ai, bi):
                    if aj is None or bj is None:
                        inner.append(None)
                    else:
                        inner.append(aj + bj)
                outer.append(inner)
        return outer

    a00 = awkward1.Array([[1.1,  2.2, 3.3],   [], [4.4, 5.5], [ 6.6, 7.7, 8.8, 9.9]])
    a01 = awkward1.Array([[1.1, None, 3.3],   [], [4.4, 5.5], [ 6.6, 7.7, 8.8, 9.9]])
    a02 = awkward1.Array([[1.1, None, 3.3],   [], [4.4, 5.5], [None, 7.7, 8.8, 9.9]])
    a10 = awkward1.Array([[1.1,  2.2, 3.3],   [],       None, [ 6.6, 7.7, 8.8, 9.9]])
    a11 = awkward1.Array([[1.1, None, 3.3],   [],       None, [ 6.6, 7.7, 8.8, 9.9]])
    a12 = awkward1.Array([[1.1, None, 3.3],   [],       None, [None, 7.7, 8.8, 9.9]])
    a20 = awkward1.Array([[1.1,  2.2, 3.3], None,       None, [ 6.6, 7.7, 8.8, 9.9]])
    a21 = awkward1.Array([[1.1, None, 3.3], None,       None, [ 6.6, 7.7, 8.8, 9.9]])
    a22 = awkward1.Array([[1.1, None, 3.3], None,       None, [None, 7.7, 8.8, 9.9]])

    b00 = awkward1.Array([[100,  200, 300],   [], [400, 500], [ 600, 700, 800, 900]])
    b01 = awkward1.Array([[100, None, 300],   [], [400, 500], [ 600, 700, 800, 900]])
    b02 = awkward1.Array([[100, None, 300],   [], [400, 500], [None, 700, 800, 900]])
    b10 = awkward1.Array([[100,  200, 300],   [],       None, [ 600, 700, 800, 900]])
    b11 = awkward1.Array([[100, None, 300],   [],       None, [ 600, 700, 800, 900]])
    b12 = awkward1.Array([[100, None, 300],   [],       None, [None, 700, 800, 900]])
    b20 = awkward1.Array([[100,  200, 300], None,       None, [ 600, 700, 800, 900]])
    b21 = awkward1.Array([[100, None, 300], None,       None, [ 600, 700, 800, 900]])
    b22 = awkward1.Array([[100, None, 300], None,       None, [None, 700, 800, 900]])

    for a in (a00, a01, a02, a10, a11, a12, a20, a21, a22):
        for b in (b00, b01, b02, b10, b11, b12, b20, b21, b22):
            assert awkward1.tolist(a + b) == add(a, b)
