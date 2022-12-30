# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak

to_list = ak.operations.to_list

empty = ak.highlevel.Array(
    ak.contents.RegularArray(
        ak.highlevel.Array([[1, 2, 3], [], [4, 5]]).layout, 0, zeros_length=0
    )
)


def test_ListOffsetArray_rpad_and_clip():
    array = ak.highlevel.Array([[1, 2, 3], [], [4, 5]])
    assert ak.operations.pad_none(array, 0, clip=True).to_list() == [
        [],
        [],
        [],
    ]

    array = ak.highlevel.Array([[1, 2, 3], [], [4, 5]])
    assert ak.operations.pad_none(array, 0).to_list() == [
        [1, 2, 3],
        [],
        [4, 5],
    ]


def test_to_ListOffsetArray64():
    assert ak.operations.from_regular(empty).to_list() == []


def test_carry():
    assert empty[[]].to_list() == []


def test_num():
    assert ak.operations.num(empty, axis=0) == 0
    assert ak.operations.num(empty, axis=1).to_list() == []
    assert ak.operations.num(empty, axis=2).to_list() == []


def test_flatten():
    assert ak.operations.flatten(empty, axis=0).to_list() == []
    assert ak.operations.flatten(empty, axis=1).to_list() == []
    assert ak.operations.flatten(empty, axis=2).to_list() == []


def test_mergeable():
    assert ak.operations.concatenate([empty, empty]).to_list() == []


def test_fillna():
    assert ak.operations.fill_none(empty, 5, axis=0).to_list() == []


def test_pad_none():
    assert ak.operations.pad_none(empty, 0, axis=0).to_list() == []
    assert ak.operations.pad_none(empty, 0, axis=1).to_list() == []
    assert ak.operations.pad_none(empty, 0, axis=2).to_list() == []

    assert ak.operations.pad_none(empty, 1, axis=0).to_list() == [None]
    assert ak.operations.pad_none(empty, 1, axis=1).to_list() == []
    assert ak.operations.pad_none(empty, 1, axis=2).to_list() == []

    assert ak.operations.pad_none(empty, 0, axis=0, clip=True).to_list() == []
    assert ak.operations.pad_none(empty, 0, axis=1, clip=True).to_list() == []
    assert ak.operations.pad_none(empty, 0, axis=2, clip=True).to_list() == []

    assert ak.operations.pad_none(empty, 1, axis=0, clip=True).to_list() == [None]
    assert ak.operations.pad_none(empty, 1, axis=1, clip=True).to_list() == []
    assert ak.operations.pad_none(empty, 1, axis=2, clip=True).to_list() == []


def test_reduce():
    assert ak.operations.sum(empty, axis=0).to_list() == []


def test_localindex():
    assert ak.operations.local_index(empty, axis=0).to_list() == []
    assert ak.operations.local_index(empty, axis=1).to_list() == []
    assert ak.operations.local_index(empty, axis=2).to_list() == []


def test_combinations():
    assert ak.operations.combinations(empty, 2, axis=0).to_list() == []
    assert ak.operations.combinations(empty, 2, axis=1).to_list() == []
    assert ak.operations.combinations(empty, 2, axis=2).to_list() == []


def test_getitem():
    with pytest.raises(IndexError):
        empty[
            0,
        ]

    jagged = ak.highlevel.Array([[]])[0:0]
    assert empty[jagged].to_list() == []
