# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_lengths_empty_regular_slices():
    d = np.arange(2 * 3).reshape(2, 3)
    e = ak.contents.NumpyArray(d)

    assert ak.to_list(d[:, []]) == ak.to_list(e[:, []]) == [[], []]
    assert ak.to_list(d[1:, []]) == ak.to_list(e[1:, []]) == [[]]

    assert d[:, []].shape == ak.to_numpy(e[:, []]).shape
    assert d[1:, []].shape == ak.to_numpy(e[1:, []]).shape

    f = ak.operations.to_regular(e, axis=1)
    assert to_list(f[:, []]) == [[], []]

    d = np.arange(5 * 7 * 11 * 13 * 17).reshape(5, 7, 11, 13, 17)
    e = ak.contents.NumpyArray(d)
    f = ak.operations.to_regular(e, axis=1)

    assert (
        ak.to_list(d[-4:, -4:, []])
        == ak.to_list(e[-4:, -4:, []])
        == ak.to_list(f[-4:, -4:, []])
        == [[[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]]
    )
    assert (
        ak.to_list(d[:, [], []])
        == ak.to_list(e[:, [], []])
        == ak.to_list(f[:, [], []])
        == [[], [], [], [], []]
    )
    assert (
        ak.to_list(d[1:4, [], []])
        == ak.to_list(e[1:4, [], []])
        == ak.to_list(f[1:4, [], []])
        == [[], [], []]
    )
    assert (
        ak.to_list(d[:, [], [], []])
        == ak.to_list(e[:, [], [], []])
        == ak.to_list(f[:, [], [], []])
        == [[], [], [], [], []]
    )
    assert (
        ak.to_list(d[1:4, [], [], []])
        == ak.to_list(e[1:4, [], [], []])
        == ak.to_list(f[1:4, [], [], []])
        == [[], [], []]
    )
    with pytest.raises(ValueError):
        _ = e[:, [], :, 0, []]

    with pytest.raises(ValueError):
        _ = e[:, [], :, []]

    with pytest.raises(ValueError):
        _ = f[:, [], :, []]

    with pytest.raises(ValueError):
        _ = e[1:4, [], -3:, []]

    with pytest.raises(ValueError):
        _ = f[1:4, [], -3:, []]

    assert (
        d[:, :, []].shape
        == ak.to_numpy(e[:, :, []]).shape
        == ak.to_numpy(f[:, :, []]).shape
    )
    assert (
        d[:, [], []].shape
        == ak.to_numpy(e[:, [], []]).shape
        == ak.to_numpy(f[:, [], []]).shape
    )
    assert (
        d[1:4, :, [], []].shape
        == ak.to_numpy(e[1:4, :, [], []]).shape
        == ak.to_numpy(f[1:4, :, [], []]).shape
    )
    assert (
        d[:, [], [], []].shape
        == ak.to_numpy(e[:, [], [], []]).shape
        == ak.to_numpy(f[:, [], [], []]).shape
    )
    assert (
        d[1:4, [], [], []].shape
        == ak.to_numpy(e[1:4, [], [], []]).shape
        == ak.to_numpy(f[1:4, [], [], []]).shape
    )

    np1 = np.ones((5, 7))
    a = ak.Array(np.ones((5, 7)))
    assert ak.to_list(np1[:, []]) == ak.to_list(a[:, []]) == [[], [], [], [], []]
    assert ak.to_list(np1[[], :]) == ak.to_list(a[[], :]) == []

    assert np1[:, []].shape == ak.to_numpy(a[:, []]).shape == (5, 0)
    assert np1[[], :].shape == ak.to_numpy(a[[], :]).shape == (0, 7)
