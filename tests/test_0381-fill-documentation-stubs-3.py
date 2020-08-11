# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test_pandas_style_constructor():
    a = awkward1.Array({"x": [1, 2, 3, 4], "y": [[1, 2, 3], [], [4, 5]]})
    assert awkward1.to_list(a) == [{"x": 1, "y": [1, 2, 3]}, {"x": 2, "y": []}, {"x": 3, "y": [4, 5]}]


pyarrow = pytest.importorskip("pyarrow")


def test_pyarrow_constructor():
    a = awkward1.Array(pyarrow.array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]))
    assert awkward1.to_list(a) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
