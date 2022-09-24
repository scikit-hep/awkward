# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401

to_list = ak.operations.to_list

empty = ak.highlevel.Array(
    ak.contents.RegularArray(
        ak.highlevel.Array([[1, 2, 3], [], [4, 5]]).layout, 0, zeros_length=0
    )
)


def test_ListOffsetArray_rpad_and_clip():
    array = ak.highlevel.Array([[1, 2, 3], [], [4, 5]])
    assert ak.operations.pad_none(array, 0, clip=True).tolist() == [
        [],
        [],
        [],
    ]

    array = ak.highlevel.Array([[1, 2, 3], [], [4, 5]])
    assert ak.operations.pad_none(array, 0).tolist() == [
        [1, 2, 3],
        [],
        [4, 5],
    ]


def test_toListOffsetArray64():
    assert ak.operations.from_regular(empty).tolist() == []


def test_carry():
    assert empty[[]].tolist() == []


def test_num():
    assert ak.operations.num(empty, axis=0) == 0
    assert ak.operations.num(empty, axis=1).tolist() == []
    assert ak.operations.num(empty, axis=2).tolist() == []


def test_flatten():
    assert ak.operations.flatten(empty, axis=0).tolist() == []
    assert ak.operations.flatten(empty, axis=1).tolist() == []
    assert ak.operations.flatten(empty, axis=2).tolist() == []


def test_mergeable():
    assert ak.operations.concatenate([empty, empty]).tolist() == []


def test_fillna():
    assert ak.operations.fill_none(empty, 5, axis=0).tolist() == []


def test_pad_none():
    assert ak.operations.pad_none(empty, 0, axis=0).tolist() == []
    assert ak.operations.pad_none(empty, 0, axis=1).tolist() == []
    assert ak.operations.pad_none(empty, 0, axis=2).tolist() == []

    assert ak.operations.pad_none(empty, 1, axis=0).tolist() == [None]
    assert ak.operations.pad_none(empty, 1, axis=1).tolist() == []
    assert ak.operations.pad_none(empty, 1, axis=2).tolist() == []

    assert ak.operations.pad_none(empty, 0, axis=0, clip=True).tolist() == []
    assert ak.operations.pad_none(empty, 0, axis=1, clip=True).tolist() == []
    assert ak.operations.pad_none(empty, 0, axis=2, clip=True).tolist() == []

    assert ak.operations.pad_none(empty, 1, axis=0, clip=True).tolist() == [None]
    assert ak.operations.pad_none(empty, 1, axis=1, clip=True).tolist() == []
    assert ak.operations.pad_none(empty, 1, axis=2, clip=True).tolist() == []


def test_reduce():
    assert ak.operations.sum(empty, axis=0).tolist() == []


def test_localindex():
    assert ak.operations.local_index(empty, axis=0).tolist() == []
    assert ak.operations.local_index(empty, axis=1).tolist() == []
    assert ak.operations.local_index(empty, axis=2).tolist() == []


def test_combinations():
    assert ak.operations.combinations(empty, 2, axis=0).tolist() == []
    assert ak.operations.combinations(empty, 2, axis=1).tolist() == []
    assert ak.operations.combinations(empty, 2, axis=2).tolist() == []


def test_getitem():
    with pytest.raises(IndexError):
        empty[
            0,
        ]

    jagged = ak.highlevel.Array([[]])[0:0]
    assert empty[jagged].tolist() == []
