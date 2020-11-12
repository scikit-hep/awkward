# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test():
    array = awkward1.Array([[[0.0, 1.1, 2.2], []], [[3.3, 4.4]], [], [[5.5], [], [6.6, 7.7, 8.8, 9.9]]])
    assert awkward1.to_list(awkward1.local_index(array, axis=0)) == [0, 1, 2, 3]
    assert awkward1.to_list(awkward1.local_index(array, axis=1)) == [[0, 1], [0], [], [0, 1, 2]]
    assert awkward1.to_list(awkward1.local_index(array, axis=2)) == [[[0, 1, 2], []], [[0, 1]], [], [[0], [], [0, 1, 2, 3]]]
    assert awkward1.to_list(awkward1.local_index(array, axis=-1)) == [[[0, 1, 2], []], [[0, 1]], [], [[0], [], [0, 1, 2, 3]]]
    assert awkward1.to_list(awkward1.local_index(array, axis=-2)) == [[0, 1], [0], [], [0, 1, 2]]
    assert awkward1.to_list(awkward1.local_index(array, axis=-3)) == [0, 1, 2, 3]
    
    assert awkward1.to_list(awkward1.zip([awkward1.local_index(array, axis=0), awkward1.local_index(array, axis=1), awkward1.local_index(array, axis=2)])) == [[[(0, 0, 0), (0, 0, 1), (0, 0, 2)], []], [[(1, 0, 0), (1, 0, 1)]], [], [[(3, 0, 0)], [], [(3, 2, 0), (3, 2, 1), (3, 2, 2), (3, 2, 3)]]]
