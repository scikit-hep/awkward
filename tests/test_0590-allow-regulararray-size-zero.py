# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


empty = ak.Array(
    ak.layout.RegularArray(ak.Array([[1, 2, 3], [], [4, 5]]).layout, 0, zeros_length=0)
)


def test_ListOffsetArray_rpad_and_clip():
    array = ak.Array([[1, 2, 3], [], [4, 5]])
    assert ak.pad_none(array, 0, clip=True).tolist() == [[], [], []]

    array = ak.Array([[1, 2, 3], [], [4, 5]])
    assert ak.pad_none(array, 0).tolist() == [[1, 2, 3], [], [4, 5]]


def test_toListOffsetArray64():
    assert ak.from_regular(empty).tolist() == []


def test_setidentities():
    empty2 = ak.Array(
        ak.layout.RegularArray(
            ak.Array([[1, 2, 3], [], [4, 5]]).layout, 0, zeros_length=0
        )
    )
    empty2.layout.setidentities()
    assert np.asarray(empty2.layout.identities).tolist() == []


def test_carry():
    assert empty[[]].tolist() == []


def test_num():
    assert ak.num(empty, axis=0) == 0
    assert ak.num(empty, axis=1).tolist() == []
    assert ak.num(empty, axis=2).tolist() == []


def test_flatten():
    assert ak.flatten(empty, axis=0).tolist() == []
    assert ak.flatten(empty, axis=1).tolist() == []
    assert ak.flatten(empty, axis=2).tolist() == []


def test_mergeable():
    assert ak.concatenate([empty, empty]).tolist() == []


def test_fillna():
    assert ak.fill_none(empty, 5, axis=0).tolist() == []


def test_pad_none():
    assert ak.pad_none(empty, 0, axis=0).tolist() == []
    assert ak.pad_none(empty, 0, axis=1).tolist() == []
    assert ak.pad_none(empty, 0, axis=2).tolist() == []

    assert ak.pad_none(empty, 1, axis=0).tolist() == [None]
    assert ak.pad_none(empty, 1, axis=1).tolist() == []
    assert ak.pad_none(empty, 1, axis=2).tolist() == []

    assert ak.pad_none(empty, 0, axis=0, clip=True).tolist() == []
    assert ak.pad_none(empty, 0, axis=1, clip=True).tolist() == []
    assert ak.pad_none(empty, 0, axis=2, clip=True).tolist() == []

    assert ak.pad_none(empty, 1, axis=0, clip=True).tolist() == [None]
    assert ak.pad_none(empty, 1, axis=1, clip=True).tolist() == []
    assert ak.pad_none(empty, 1, axis=2, clip=True).tolist() == []


def test_reduce():
    assert ak.sum(empty, axis=0).tolist() == []


def test_localindex():
    assert ak.local_index(empty, axis=0).tolist() == []
    assert ak.local_index(empty, axis=1).tolist() == []
    assert ak.local_index(empty, axis=2).tolist() == []


def test_combinations():
    assert ak.combinations(empty, 2, axis=0).tolist() == []
    assert ak.combinations(empty, 2, axis=1).tolist() == []
    assert ak.combinations(empty, 2, axis=2).tolist() == []


def test_getitem():
    with pytest.raises(ValueError):
        empty[
            0,
        ]

    jagged = ak.Array([[]])[0:0]
    assert empty[jagged].tolist() == []
