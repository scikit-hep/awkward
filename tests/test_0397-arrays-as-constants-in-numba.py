# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1

numba = pytest.importorskip("numba")


def test_ArrayBuilder():
    builder = awkward1.ArrayBuilder()
    assert sys.getrefcount(builder._layout) == 3

    @numba.njit
    def f():
        builder.append(1)
        builder.append(2)
        builder.append(3)
        return builder, builder

    @numba.njit
    def g():
        builder.append(1)
        builder.append(2)
        builder.append(3)

    b, c = f()
    assert b.snapshot().tolist() == [1, 2, 3]
    assert c.snapshot().tolist() == [1, 2, 3]
    assert builder.snapshot().tolist() == [1, 2, 3]

    assert sys.getrefcount(builder._layout) == 5

    g()
    assert b.snapshot().tolist() == [1, 2, 3, 1, 2, 3]
    assert c.snapshot().tolist() == [1, 2, 3, 1, 2, 3]
    assert builder.snapshot().tolist() == [1, 2, 3, 1, 2, 3]

    assert sys.getrefcount(builder._layout) == 5

    del b._layout
    assert sys.getrefcount(builder._layout) == 4

    del c._layout
    assert sys.getrefcount(builder._layout) == 3
