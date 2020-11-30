# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy as np
import awkward1 as ak


def test_iterator():
    content = ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3]))
    offsets = ak.layout.Index32(np.array([0, 2, 2, 3], "i4"))
    array = ak.layout.ListOffsetArray32(offsets, content)
    assert list(content) == [1.1, 2.2, 3.3]
    assert [np.asarray(x).tolist() for x in array] == [[1.1, 2.2], [], [3.3]]


def test_refcount():
    content = ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3]))
    offsets = ak.layout.Index32(np.array([0, 2, 2, 3], "i4"))
    array = ak.layout.ListOffsetArray32(offsets, content)

    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)

    iter1 = iter(content)
    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)
    x1 = next(iter1)
    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)

    iter2 = iter(array)
    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)
    x2 = next(iter2)
    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)

    del iter1
    del x1
    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)

    del iter2
    del x2
    assert (sys.getrefcount(content), sys.getrefcount(array)) == (2, 2)
