# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

numba = pytest.importorskip("numba")

def test_unmasked():
    @numba.njit
    def find_it(array):
        for i in range(3):
            item = array[i]
            if item is None:
                pass
            elif item.x == 3:
                return item
        return None

    content = awkward1.Array([{"x": 1}, {"x": 2}, {"x": 3}]).layout
    unmasked = awkward1.layout.UnmaskedArray(content)
    array = awkward1.Array(unmasked)
    assert awkward1.to_list(find_it(array)) == {"x": 3}

# def test_indexedoption():
#     @numba.njit
#     def find_it(array):
#         for i in range(3):
#             item = array[i]
#             if item is None:
#                 pass
#             elif item.x == 3:
#                 return item
#         return None

#     array = awkward1.Array([{"x": 1}, {"x": 2}, None, {"x": 3}])
#     assert awkward1.to_list(find_it(array)) == {"x": 3}
