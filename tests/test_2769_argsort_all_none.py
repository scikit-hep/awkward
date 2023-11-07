# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest  # noqa: F401

import awkward as ak


def test_all_none():
    result = ak.argsort([None, None, None])
    assert ak.is_valid(result)
    assert result.to_list() == [0, 1, 2]
    assert isinstance(result.layout, ak.contents.NumpyArray)
    assert result.type == ak.types.ArrayType(ak.types.NumpyType("int64"), 3)


def test_mixed_none_local():
    result = ak.argsort([None, 1, None, 2, 0, None, -1])
    assert ak.is_valid(result)
    assert result.to_list() == [6, 4, 1, 3, 0, 2, 5]
    assert result.type == ak.types.ArrayType(ak.types.NumpyType("int64"), 7)


def test_mixed_none_2d_local():
    result = ak.argsort(
        [
            [None, 1, None, 0, None, None, -1],
            None,
            [None, 2, None, 2, 0, None, -2],
        ],
        axis=1,
    )
    assert ak.is_valid(result)
    assert result.to_list() == [[6, 3, 1, 0, 2, 4, 5], None, [6, 4, 1, 3, 0, 2, 5]]
    assert result.type == ak.types.ArrayType(
        ak.types.OptionType(ak.types.ListType(ak.types.NumpyType("int64"))), 3
    )


def test_mixed_none_2d_nonlocal():
    result = ak.argsort(
        [
            [None, 1, None, 0, None, None, -1],
            [None, 2, None, 2, 0, None, -2],
        ],
        axis=0,
    )
    assert ak.is_valid(result)
    assert result.to_list() == [[0, 0, 0, 0, 1, 0, 1], [1, 1, 1, 1, 0, 1, 0]]
    assert result.type == ak.types.ArrayType(
        ak.types.ListType(ak.types.NumpyType("int64")), 2
    )
