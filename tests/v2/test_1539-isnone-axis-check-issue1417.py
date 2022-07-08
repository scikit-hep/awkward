# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test_1417issue_is_none_check_axis():
    array = ak._v2.Array([[[1, 2, None], None], None])

    assert to_list(ak._v2.is_none(array, axis=0)) == [False, True]
    assert to_list(ak._v2.is_none(array, axis=1)) == [[False, True], None]
    assert to_list(ak._v2.is_none(array, axis=2)) == [
        [[False, False, True], None],
        None,
    ]

    with pytest.raises(ValueError):
        ak._v2.is_none(array, axis=3)

    assert str(ak._v2.type(ak._v2.is_none(array, axis=0)[0])) == "bool"

    array = ak._v2.Array([[[[[1], [3]], [1]]], []])

    assert to_list(ak._v2.is_none(array, axis=2)) == [[[False, False]], []]
