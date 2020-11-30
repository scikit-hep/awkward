# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak

numexpr = pytest.importorskip("numexpr")


def test_numexpr():
    a = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)
    b = ak.Array([100, 200, 300], check_valid=True)
    assert ak.to_list(ak.numexpr.evaluate("a + b")) == [
        [101.1, 102.2, 103.3],
        [],
        [304.4, 305.5],
    ]
    a = [1, 2, 3]
    assert ak.to_list(ak.numexpr.re_evaluate()) == [101, 202, 303]


def test_broadcast_arrays():
    a = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)
    b = ak.Array([100, 200, 300], check_valid=True)

    out = ak.broadcast_arrays(a, b)
    assert ak.to_list(out[0]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert ak.to_list(out[1]) == [[100, 100, 100], [], [300, 300]]
