# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import itertools

import pytest
import numpy

import awkward1

def test():
    assert awkward1.to_list(awkward1.prod(awkward1.Array([[[2, 3, 5]], [[7], [11]], [[]]]), axis=-1)) == [[30], [7, 11], [1]]

    assert awkward1.to_list(awkward1.prod(awkward1.Array([[[2, 3, 5]], [[7], [11]], []]), axis=-1)) == [[30], [7, 11], []]
