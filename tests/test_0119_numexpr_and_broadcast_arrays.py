# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_broadcast_arrays():
    a = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)
    b = ak.Array([100, 200, 300], check_valid=True)

    out = ak.operations.broadcast_arrays(a, b)
    assert to_list(out[0]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert to_list(out[1]) == [[100, 100, 100], [], [300, 300]]


numexpr = pytest.importorskip("numexpr")


def test_numexpr():
    # NumExpr's interface pulls variables from the surrounding scope,
    # so these F841 "unused variables" actually are used.

    a = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)
    b = ak.Array([100, 200, 300], check_valid=True)  # noqa: F841
    assert to_list(ak._connect.numexpr.evaluate("a + b")) == [
        [101.1, 102.2, 103.3],
        [],
        [304.4, 305.5],
    ]
    a = [1, 2, 3]  # noqa: F841
    assert to_list(ak._connect.numexpr.re_evaluate()) == [101, 202, 303]
