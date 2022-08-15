# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test_combinations():
    two = ak._v2.Array(["aa", "bb", "cc", "dd"])
    with pytest.raises(ValueError):
        ak._v2.operations.combinations(two, 2, axis=1)

    two = ak._v2.Array([["aa", "bb"], ["cc"], [], ["dd"]])
    assert to_list(ak._v2.operations.combinations(two, 2, axis=1)) == [
        [("aa", "bb")],
        [],
        [],
        [],
    ]
    with pytest.raises(ValueError):
        ak._v2.operations.combinations(two, 2, axis=2)


def test_cartesian():
    one = ak._v2.Array([1, 2, 3, 4])
    two = ak._v2.Array(["aa", "bb", "cc", "dd"])
    with pytest.raises(ValueError):
        ak._v2.operations.cartesian([one, two], axis=1)

    two = ak._v2.Array([["aa", "bb"], ["cc"], [], ["dd"]])
    assert to_list(ak._v2.operations.cartesian([one, two], axis=1)) == [
        [(1, "aa"), (1, "bb")],
        [(2, "cc")],
        [],
        [(4, "dd")],
    ]
    with pytest.raises(ValueError):
        ak._v2.operations.cartesian([one, two], axis=2)
