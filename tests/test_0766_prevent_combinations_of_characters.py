# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_combinations():
    two = ak.Array(["aa", "bb", "cc", "dd"])
    with pytest.raises(ValueError):
        ak.operations.combinations(two, 2, axis=1)

    two = ak.Array([["aa", "bb"], ["cc"], [], ["dd"]])
    assert to_list(ak.operations.combinations(two, 2, axis=1)) == [
        [("aa", "bb")],
        [],
        [],
        [],
    ]
    with pytest.raises(ValueError):
        ak.operations.combinations(two, 2, axis=2)


def test_cartesian():
    one = ak.Array([1, 2, 3, 4])
    two = ak.Array(["aa", "bb", "cc", "dd"])
    with pytest.raises(ValueError):
        ak.operations.cartesian([one, two], axis=1)

    two = ak.Array([["aa", "bb"], ["cc"], [], ["dd"]])
    assert to_list(ak.operations.cartesian([one, two], axis=1)) == [
        [(1, "aa"), (1, "bb")],
        [(2, "cc")],
        [],
        [(4, "dd")],
    ]
    with pytest.raises(ValueError):
        ak.operations.cartesian([one, two], axis=2)
