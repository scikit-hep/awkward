# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_singletons():
    array = awkward1.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])
    assert awkward1.to_list(awkward1.singletons(array)) == [[1.1], [2.2], [], [3.3], [], [], [4.4], [5.5]]
