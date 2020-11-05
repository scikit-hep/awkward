# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test():
    one = awkward1.Array([999, 123, 1, 2, 3, 4, 5])
    two = awkward1.Array([999])[:0]
    three = awkward1.Array([])

    assert awkward1.to_list(one[[None, None]]) == [None, None]
    assert awkward1.to_list(one[[None, 0, None]]) == [None, 999, None]

    assert awkward1.to_list(two[[None, None]]) == [None, None]
    assert awkward1.to_list(two[[None, None, None]]) == [None, None, None]

    assert awkward1.to_list(three[[None, None]]) == [None, None]
    assert awkward1.to_list(three[[None, None, None]]) == [None, None, None]

    array = awkward1.Array([[[0, 1, 2], []], [[], [3, 4]], [[5], [6, 7, 8, 9]]])
    assert awkward1.to_list(array[:, [None, 1, None]]) == [[None, [], None], [None, [3, 4], None], [None, [6, 7, 8, 9], None]]
    assert awkward1.to_list(array[:2, [None, 1, None]]) == [[None, [], None], [None, [3, 4], None]]
    assert awkward1.to_list(array[1:, [None, 1, None]]) == [[None, [3, 4], None], [None, [6, 7, 8, 9], None]]
    assert awkward1.to_list(array[:0, [None, 1, None]]) == []
