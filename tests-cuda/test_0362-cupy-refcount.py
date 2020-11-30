# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import cupy as cp  # noqa: F401
import awkward as ak  # noqa: F401


def test_cupy_refcount():
    o = cp.arange(10)
    i = ak.layout.Index64.from_cupy(o)
    assert sys.getrefcount(o) == 3
    i2 = ak.layout.Index64.from_cupy(o)
    assert sys.getrefcount(o) == 4
    i3 = ak.layout.Index64.from_cupy(o)
    assert sys.getrefcount(o) == 5
    i4 = ak.layout.Index64.from_cupy(o)
    assert sys.getrefcount(o) == 6
    i5 = ak.layout.Index64.from_cupy(o)
    assert sys.getrefcount(o) == 7
    i6 = ak.layout.Index64.from_cupy(o)
    assert sys.getrefcount(o) == 8
    i7 = ak.layout.Index64.from_cupy(o)
    assert sys.getrefcount(o) == 9
    i8 = ak.layout.Index64.from_cupy(o)
    assert sys.getrefcount(o) == 10
    i9 = ak.layout.Index64.from_cupy(o)
    assert sys.getrefcount(o) == 11

    del i9
    assert sys.getrefcount(o) == 10
    del i8
    assert sys.getrefcount(o) == 9
    del i7
    assert sys.getrefcount(o) == 8
    del i6
    assert sys.getrefcount(o) == 7
    del i5
    assert sys.getrefcount(o) == 6
    del i4
    assert sys.getrefcount(o) == 5
    del i3
    assert sys.getrefcount(o) == 4
    del i2
    assert sys.getrefcount(o) == 3
    del i
    assert sys.getrefcount(o) == 2
