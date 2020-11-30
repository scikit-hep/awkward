# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import itertools

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward1 as ak  # noqa: F401


def test_refcount():
    o = np.arange(10, dtype="i4")
    c = np.arange(12).reshape(3, 4)

    for order in itertools.permutations(["del i, n", "del l", "del l2"]):
        i = ak.layout.Index32(o)
        n = ak.layout.NumpyArray(c)
        l = ak.layout.ListOffsetArray32(i, n)
        l2 = ak.layout.ListOffsetArray32(i, l)

        for statement in order:
            assert sys.getrefcount(o), sys.getrefcount(c) == (3, 3)
            exec(statement)
            assert sys.getrefcount(o), sys.getrefcount(c) == (2, 2)
