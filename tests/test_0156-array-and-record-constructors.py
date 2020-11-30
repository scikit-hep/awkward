# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak

def test_record():
    assert ak.to_list(ak.Record({"x": 1, "y": 2.2})) == {"x": 1, "y": 2.2}

def test_fromiter():
    array = ak.Array([np.array([1, 2, 3]), np.array([4, 5, 6, 7])])

    assert str(ak.type(array)) == "2 * var * int64"
    assert ak.to_list(array) == [[1, 2, 3], [4, 5, 6, 7]]
