# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak


def test_regular_string_mixed_invalid():
    strings = ak.to_regular([["abc", "efg"]], axis=2)
    numbers = ak.to_regular([[[1, 2, 3], [3, 4, 3]]], axis=2)

    with pytest.raises(
        ValueError,
        match=r"cannot broadcast RegularArray of length 2 with NumpyArray of length 6",
    ):
        ak.broadcast_arrays(strings, numbers, right_broadcast=False)


def test_regular_string_mixed_valid():
    strings = ak.to_regular([["abc", "efg"]], axis=2)
    numbers = ak.to_regular([[[1], [3]]], axis=2)

    x, y = ak.broadcast_arrays(strings, numbers, right_broadcast=False)
    assert x.tolist() == [[["abc"], ["efg"]]]
    assert y.tolist() == [[[1], [3]]]


def test_regular_string_mixed_below():
    strings = ak.to_regular([["abc", "efg"]], axis=2)
    numbers = ak.to_regular([[1, 6], [3, 7]], axis=1)

    x, y = ak.broadcast_arrays(strings, numbers, right_broadcast=False)
    assert x.tolist() == [["abc", "efg"], ["abc", "efg"]]
    assert y.tolist() == [[1, 6], [3, 7]]


def test_regular_string_string_valid():
    strings = ak.to_regular([["abc", "efg"]], axis=2)
    numbers = ak.to_regular([[["ab"], ["bc", "de"]]], axis=3)

    x, y = ak.broadcast_arrays(strings, numbers, right_broadcast=False)
    assert x.tolist() == [[["abc"], ["efg", "efg"]]]
    assert y.tolist() == [[["ab"], ["bc", "de"]]]
