# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test_broadcast_arrays():
    a = ak._v2.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)
    b = ak._v2.Array([100, 200, 300], check_valid=True)

    out = ak._v2.operations.broadcast_arrays(a, b)
    assert to_list(out[0]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert to_list(out[1]) == [[100, 100, 100], [], [300, 300]]


numexpr = pytest.importorskip("numexpr")


def test_numexpr():
    # NumExpr's interface pulls variables from the surrounding scope,
    # so these F841 "unused variables" actually are used.

    a = ak._v2.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)  # noqa: F841
    b = ak._v2.Array([100, 200, 300], check_valid=True)  # noqa: F841
    assert to_list(ak._v2._connect.numexpr.evaluate("a + b")) == [
        [101.1, 102.2, 103.3],
        [],
        [304.4, 305.5],
    ]
    a = [1, 2, 3]  # noqa: F841
    assert to_list(ak._v2._connect.numexpr.re_evaluate()) == [101, 202, 303]
