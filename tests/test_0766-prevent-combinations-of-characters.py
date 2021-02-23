# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    one = ak.Array([1, 2, 3, 4])
    two = ak.Array(["aa", "bb", "cc", "dd"])
    with pytest.raises(ValueError):
        ak.cartesian([one, two], axis=1)
    with pytest.raises(ValueError):
        ak.combinations(two, 2, axis=1)

    two = ak.Array([["aa", "bb"], ["cc"], [], ["dd"]])
    assert ak.to_list(ak.cartesian([one, two], axis=1)) == [
        [(1, "aa"), (1, "bb")],
        [(2, "cc")],
        [],
        [(4, "dd")],
    ]
    assert ak.to_list(ak.combinations(two, 2, axis=1)) == [[("aa", "bb")], [], [], []]

    with pytest.raises(ValueError):
        ak.cartesian([one, two], axis=2)
    with pytest.raises(ValueError):
        ak.combinations(two, 2, axis=2)
