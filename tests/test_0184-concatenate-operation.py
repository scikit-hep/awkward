# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_listoffsetarray_merge():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 9]))
    listoffsetarray1 = awkward1.layout.ListOffsetArray64(offsets1, content1)

    assert awkward1.to_list(listoffsetarray1) == [[1, 2, 3], [], [4, 5], [6, 7, 8, 9]]

    content2 = awkward1.layout.NumpyArray(numpy.array([100, 200, 300, 400, 500]))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 2, 4, 4, 5]))
    listoffsetarray2 = awkward1.layout.ListOffsetArray64(offsets2, content2)

    assert awkward1.to_list(listoffsetarray2) == [[100, 200], [300, 400], [], [500]]
    assert awkward1.to_list(listoffsetarray1.merge(listoffsetarray2, 0)) == [[1, 2, 3], [], [4, 5], [6, 7, 8, 9], [100, 200], [300, 400], [], [500]]
    # FIXME: assert awkward1.to_list(listoffsetarray1.merge(listoffsetarray2, 1)) == [[1, 2, 3, 100, 200], [300, 400], [4, 5], [6, 7, 8, 9, 500]]
