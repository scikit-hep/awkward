# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import sys

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2


def test_iterator():
    content = ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3]))
    offsets = ak.layout.Index32(np.array([0, 2, 2, 3], "i4"))
    array = ak.layout.ListOffsetArray32(offsets, content)
    content2 = v1_to_v2(content)
    array2 = v1_to_v2(array)

    assert list(content2) == [1.1, 2.2, 3.3]
    assert [np.asarray(x).tolist() for x in array2] == [[1.1, 2.2], [], [3.3]]

    assert list(content) == list(content)
    assert [np.asarray(x).tolist() for x in array2] == [
        np.asarray(x).tolist() for x in array
    ]


def test_refcount():
    content = ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3]))
    offsets = ak.layout.Index32(np.array([0, 2, 2, 3], "i4"))
    array = ak.layout.ListOffsetArray32(offsets, content)
    content2 = v1_to_v2(content)
    array2 = v1_to_v2(array)

    assert (sys.getrefcount(content2), sys.getrefcount(array2)) == (2, 2)
    iter1 = iter(content2)

    # assert (sys.getrefcount(content2), sys.getrefcount(array2)) == (2, 2)
    x1 = next(iter1)
    # assert (sys.getrefcount(content2), sys.getrefcount(array2)) == (2, 2)

    # iter2 = iter(array2)
    # assert (sys.getrefcount(content2), sys.getrefcount(array2)) == (2, 2)
    # x2 = next(iter2)
    # assert (sys.getrefcount(content2), sys.getrefcount(array2)) == (2, 2)

    # del iter1
    del x1
    # assert (sys.getrefcount(content2), sys.getrefcount(array2)) == (2, 2)

    # del iter2
    # del x2
    # assert (sys.getrefcount(content2), sys.getrefcount(array2)) == (2, 2)
