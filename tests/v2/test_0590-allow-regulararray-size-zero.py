# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.convert.to_list

empty = ak._v2.highlevel.Array(
    ak._v2.contents.RegularArray(
        ak._v2.highlevel.Array([[1, 2, 3], [], [4, 5]]).layout, 0, zeros_length=0
    )
)


@pytest.mark.skip(
    reason="FIXME: ak._v2.operations.structure.pad_none must be implemented"
)
def test_ListOffsetArray_rpad_and_clip():
    array = ak._v2.highlevel.Array([[1, 2, 3], [], [4, 5]])
    assert ak.pad_none(array, 0, clip=True).tolist() == [[], [], []]

    array = ak._v2.highlevel.Array([[1, 2, 3], [], [4, 5]])
    assert ak.pad_none(array, 0).tolist() == [[1, 2, 3], [], [4, 5]]


def test_toListOffsetArray64():
    assert ak._v2.operations.structure.from_regular(empty).tolist() == []


@pytest.mark.skip(reason="FIXME: assert failing")
def test_setidentities():
    empty2 = ak._v2.highlevel.Array(
        ak._v2.contents.RegularArray(
            ak._v2.highlevel.Array([[1, 2, 3], [], [4, 5]]).layout, 0, zeros_length=0
        )
    )

    assert np.asarray(empty2.layout.identifier).tolist() == []


def test_carry():
    assert empty[[]].tolist() == []


@pytest.mark.skip(reason="FIXME: ak.num must be implemented")
def test_num():
    assert ak.num(empty, axis=0) == 0
    assert ak.num(empty, axis=1).tolist() == []
    assert ak.num(empty, axis=2).tolist() == []


@pytest.mark.skip(reason="FIXME: ak.flatten must be merged")
def test_flatten():
    assert ak.flatten(empty, axis=0).tolist() == []
    assert ak.flatten(empty, axis=1).tolist() == []
    assert ak.flatten(empty, axis=2).tolist() == []


def test_mergeable():
    assert ak._v2.operations.structure.concatenate([empty, empty]).tolist() == []


def test_fillna():
    assert ak._v2.operations.structure.fill_none(empty, 5, axis=0).tolist() == []


@pytest.mark.skip(reason="FIXME: ak.pad_none must be merged")
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
    assert ak._v2.operations.reducers.sum(empty, axis=0).tolist() == []


def test_localindex():
    assert ak._v2.operations.structure.local_index(empty, axis=0).tolist() == []
    assert ak._v2.operations.structure.local_index(empty, axis=1).tolist() == []
    assert ak._v2.operations.structure.local_index(empty, axis=2).tolist() == []


@pytest.mark.skip(reason="FIXME: ak.combinations must be merged")
def test_combinations():
    assert ak.combinations(empty, 2, axis=0).tolist() == []
    assert ak.combinations(empty, 2, axis=1).tolist() == []
    assert ak.combinations(empty, 2, axis=2).tolist() == []


def test_getitem():
    with pytest.raises(IndexError):
        empty[
            0,
        ]

    jagged = ak._v2.highlevel.Array([[]])[0:0]
    assert empty[jagged].tolist() == []
