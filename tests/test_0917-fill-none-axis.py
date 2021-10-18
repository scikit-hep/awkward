# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


array = ak.Array([[None, 2], None, [4, None]])


def test_fill_none_axis_0():
    filled = ak.fill_none(array, 10, axis=0)
    assert ak.to_list(filled) == [[None, 2], 10, [4, None]]


def test_fill_none_axis_1():
    filled = ak.fill_none(array, 10, axis=1)
    assert ak.to_list(filled) == [[10, 2], None, [4, 10]]


def test_fill_none_axis_none():
    filled = ak.fill_none(array, 10, axis=None)
    assert ak.to_list(filled) == [[10, 2], 10, [4, 10]]
