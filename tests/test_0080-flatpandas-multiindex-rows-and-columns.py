# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

pandas = pytest.importorskip("pandas")

def test():
    array = awkward1.Array([[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]], None, [[], None, [6.6, None, 8.8, 9.9]]])
    array = awkward1.Array([[[{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}], [], [{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}]], [], [[{"x": 5.5, "y": [5, 5, 5, 5, 5]}]]])
    array = awkward1.Array([[[{"x": 0.0, "y": 0}, {"x": 1.1, "y": 1}, {"x": 2.2, "y": 2}], [], [{"x": 3.3, "y": 3}, {"x": 4.4, "y": 4}]], [], [[{"x": 5.5, "y": 5}]]])
    array = awkward1.Array([[[{"x": 0.0, "y": {"z": 0}}, {"x": 1.1, "y": {"z": 1}}, {"x": 2.2, "y": {"z": 2}}], [], [{"x": 3.3, "y": {"z": 3}}, {"x": 4.4, "y": {"z": 4}}]], [], [[{"x": 5.5, "y": {"z": 5}}]]])

    # for table in awkward1.pandas.multiindex(array):
    #     print(table)

    # raise Exception
