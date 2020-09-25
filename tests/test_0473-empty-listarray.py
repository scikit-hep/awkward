# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test_empty_listarray():
    a = awkward1.Array(
        awkward1.layout.ListArray64(
            awkward1.layout.Index64(numpy.array([], dtype=numpy.int64)),
            awkward1.layout.Index64(numpy.array([], dtype=numpy.int64)),
            awkward1.layout.NumpyArray(numpy.array([])),
        )
    )
    assert awkward1.to_list(a * 3) == []
