# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401
import numpy as np  # noqa: F401

to_list = ak._v2.operations.to_list


def test_lengths_empty_regular_slices():
    d = np.arange(2 * 3).reshape(2, 3)
    e = ak._v2.contents.NumpyArray(d)

    assert ak._v2.to_list(d[:, []]) == ak._v2.to_list(e[:, []]) == [[], []]
    assert ak._v2.to_list(d[1:, []]) == ak._v2.to_list(e[1:, []]) == [[]]

    assert d[:, []].shape == ak._v2.to_numpy(e[:, []]).shape
    assert d[1:, []].shape == ak._v2.to_numpy(e[1:, []]).shape

    f = ak._v2.operations.to_regular(e, axis=1)
    assert to_list(f[:, []]) == [[], []]

    d = np.arange(5 * 7 * 11 * 13 * 17).reshape(5, 7, 11, 13, 17)
    e = ak._v2.contents.NumpyArray(d)
    f = ak._v2.operations.to_regular(e, axis=1)

    assert (
        ak._v2.to_list(d[-4:, -4:, []])
        == ak._v2.to_list(e[-4:, -4:, []])
        == ak._v2.to_list(f[-4:, -4:, []])
        == [[[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]]
    )
    assert (
        ak._v2.to_list(d[:, [], []])
        == ak._v2.to_list(e[:, [], []])
        == ak._v2.to_list(f[:, [], []])
        == [[], [], [], [], []]
    )
    assert (
        ak._v2.to_list(d[1:4, [], []])
        == ak._v2.to_list(e[1:4, [], []])
        == ak._v2.to_list(f[1:4, [], []])
        == [[], [], []]
    )
    assert (
        ak._v2.to_list(d[:, [], [], []])
        == ak._v2.to_list(e[:, [], [], []])
        == ak._v2.to_list(f[:, [], [], []])
        == [[], [], [], [], []]
    )
    assert (
        ak._v2.to_list(d[1:4, [], [], []])
        == ak._v2.to_list(e[1:4, [], [], []])
        == ak._v2.to_list(f[1:4, [], [], []])
        == [[], [], []]
    )
    assert (
        ak._v2.to_list(d[:, [], :, []])
        == ak._v2.to_list(e[:, [], :, []])
        == ak._v2.to_list(f[:, [], :, []])
        == []
    )
    assert (
        ak._v2.to_list(d[1:4, [], -3:, []])
        == ak._v2.to_list(e[1:4, [], -3:, []])
        == ak._v2.to_list(f[1:4, [], -3:, []])
        == []
    )

    assert (
        d[:, :, []].shape
        == ak._v2.to_numpy(e[:, :, []]).shape
        == ak._v2.to_numpy(f[:, :, []]).shape
    )
    assert (
        d[:, [], []].shape
        == ak._v2.to_numpy(e[:, [], []]).shape
        == ak._v2.to_numpy(f[:, [], []]).shape
    )
    assert (
        d[1:4, :, [], []].shape
        == ak._v2.to_numpy(e[1:4, :, [], []]).shape
        == ak._v2.to_numpy(f[1:4, :, [], []]).shape
    )
    assert (
        d[:, [], [], []].shape
        == ak._v2.to_numpy(e[:, [], [], []]).shape
        == ak._v2.to_numpy(f[:, [], [], []]).shape
    )
    assert (
        d[1:4, [], [], []].shape
        == ak._v2.to_numpy(e[1:4, [], [], []]).shape
        == ak._v2.to_numpy(f[1:4, [], [], []]).shape
    )
    assert (
        d[:, [], :, []].shape
        == ak._v2.to_numpy(e[:, [], :, []]).shape
        == ak._v2.to_numpy(f[:, [], :, []]).shape
    )
    assert (
        d[1:4, [], -3:, []].shape
        == ak._v2.to_numpy(e[1:4, [], -3:, []]).shape
        == ak._v2.to_numpy(f[1:4, [], -3:, []]).shape
    )
