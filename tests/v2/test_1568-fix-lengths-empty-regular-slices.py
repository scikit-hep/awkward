# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test_1999():
    a = ak._v2.Array([[1, 2, 3], [4, 5, 6]])
    assert to_list(a[:, []]) == [[], []]

    b = ak._v2.operations.to_regular(a, axis=1)
    assert to_list(b[:, []]) == [[], []]
