# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test():
    data = awkward1.Array([[7, 5, 7], [], [2], [8, 2]])
    assert awkward1.to_list(awkward1.sort(data)) == [[5, 7, 7], [], [2], [2, 8]]

    index = awkward1.argsort(data)
    assert awkward1.to_list(data[index]) == [[5, 7, 7], [], [2], [2, 8]]
