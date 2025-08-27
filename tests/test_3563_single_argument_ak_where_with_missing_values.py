# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward.operations import to_list


def test():
    onedim = ak.Array([1, 2, 3, None, 4, 5])
    result = ak.where(onedim)
    assert to_list(result[0]) == [0, 1, 2, 4, 5]
    assert to_list(result[0]) == ak.to_numpy(onedim).nonzero()[0].tolist()

    onedim = ak.Array([True, False, True, None, False])
    result = ak.where(onedim)
    assert to_list(result[0]) == [0, 2]
    assert to_list(result[0]) == ak.to_numpy(onedim).nonzero()[0].tolist()

    twodim = ak.Array([[1, 2, 3], [None, 4, 5]])
    result = ak.where(twodim)
    assert to_list(result[0]) == [0, 0, 0, 1, 1]
    assert to_list(result[0]) == ak.to_numpy(twodim).nonzero()[0].tolist()
    assert to_list(result[1]) == [0, 1, 2, 1, 2]
    assert to_list(result[1]) == ak.to_numpy(twodim).nonzero()[1].tolist()

    twodim = ak.Array([[True, False, True], [None, False, True]])
    result = ak.where(twodim)
    assert to_list(result[0]) == [0, 0, 1]
    assert to_list(result[0]) == ak.to_numpy(twodim).nonzero()[0].tolist()
    assert to_list(result[1]) == [0, 2, 2]
    assert to_list(result[1]) == ak.to_numpy(twodim).nonzero()[1].tolist()

    threedim = ak.Array([[[1, 2], [3, None]], [[4, 5], [None, None]]])
    result = ak.where(threedim)
    assert to_list(result[0]) == [0, 0, 0, 1, 1]
    assert to_list(result[0]) == ak.to_numpy(threedim).nonzero()[0].tolist()
    assert to_list(result[1]) == [0, 0, 1, 0, 0]
    assert to_list(result[1]) == ak.to_numpy(threedim).nonzero()[1].tolist()
    assert to_list(result[2]) == [0, 1, 0, 0, 1]
    assert to_list(result[2]) == ak.to_numpy(threedim).nonzero()[2].tolist()

    threedim = ak.Array([[[True, False], [True, None]], [[False, True], [None, None]]])
    result = ak.where(threedim)
    assert to_list(result[0]) == [0, 0, 1]
    assert to_list(result[0]) == ak.to_numpy(threedim).nonzero()[0].tolist()
    assert to_list(result[1]) == [0, 1, 0]
    assert to_list(result[1]) == ak.to_numpy(threedim).nonzero()[1].tolist()
    assert to_list(result[2]) == [0, 0, 1]
    assert to_list(result[2]) == ak.to_numpy(threedim).nonzero()[2].tolist()
