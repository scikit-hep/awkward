# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward.operations import to_list


def test_ak_where_single_argument_with_missing_values():
    onedim = ak.Array([1, 2, 3, None, 4, 5])
    result = ak.where(onedim)
    assert to_list(result[0]) == [0, 1, 2, 4, 5]

    onedim = ak.Array([True, False, True, None, False])
    result = ak.where(onedim)
    assert to_list(result[0]) == [0, 2]

    twodim = ak.Array([[1, 2, 3], [None, 4, 5]])
    result = ak.where(twodim)
    assert to_list(result[0]) == [0, 0, 0, 1, 1]
    assert to_list(result[1]) == [0, 1, 2, 1, 2]

    twodim = ak.Array([[True, False, True], [None, False, True]])
    result = ak.where(twodim)
    assert to_list(result[0]) == [0, 0, 1]
    assert to_list(result[1]) == [0, 2, 2]

    threedim = ak.Array([[[1, 2], [3, None]], [[4, 5], [None, None]]])
    result = ak.where(threedim)
    assert to_list(result[0]) == [0, 0, 0, 1, 1]
    assert to_list(result[1]) == [0, 0, 1, 0, 0]
    assert to_list(result[2]) == [0, 1, 0, 0, 1]

    threedim = ak.Array([[[True, False], [True, None]], [[False, True], [None, None]]])
    result = ak.where(threedim)
    assert to_list(result[0]) == [0, 0, 1]
    assert to_list(result[1]) == [0, 1, 0]
    assert to_list(result[2]) == [0, 0, 1]
