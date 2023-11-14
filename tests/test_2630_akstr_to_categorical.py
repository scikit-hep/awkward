# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

pytest.importorskip("pyarrow")


def test_to_categorical_nested():
    array = ak.Array([["one", "two", "three"], [], ["one", "two"], ["three"]])
    assert not ak.operations.ak_is_categorical.is_categorical(array)
    categorical = ak.str.to_categorical(array)
    assert ak.operations.ak_is_categorical.is_categorical(categorical)
    assert ak.to_list(array) == categorical.to_list()
    not_categorical = ak.operations.ak_from_categorical.from_categorical(categorical)
    assert not ak.operations.ak_is_categorical.is_categorical(not_categorical)
    assert ak.operations.ak_categories.categories(categorical).to_list() == [
        "one",
        "two",
        "three",
    ]


def test_to_categorical():
    array = ak.Array(
        ["one", "two", "three", "one", "two", "three", "one", "two", "three"]
    )
    assert not ak.operations.ak_is_categorical.is_categorical(array)
    categorical = ak.str.to_categorical(array)
    assert ak.operations.ak_is_categorical.is_categorical(categorical)
    assert ak.to_list(array) == categorical.to_list()
    assert ak.to_list(categorical.layout.content) == ["one", "two", "three"]
    not_categorical = ak.operations.ak_from_categorical.from_categorical(categorical)
    assert not ak.operations.ak_is_categorical.is_categorical(not_categorical)
    assert ak.operations.ak_categories.categories(categorical).to_list() == [
        "one",
        "two",
        "three",
    ]


def test_to_categorical_none():
    array = ak.Array(
        [
            "one",
            "two",
            "three",
            None,
            "one",
            "two",
            "three",
            None,
            "one",
            "two",
            "three",
            None,
        ]
    )
    assert not ak.operations.ak_is_categorical.is_categorical(array)
    categorical = ak.str.to_categorical(array)
    assert ak.operations.ak_is_categorical.is_categorical(categorical)
    assert ak.to_list(array) == categorical.to_list()
    assert ak.to_list(categorical.layout.content) == ["one", "two", "three"]
    not_categorical = ak.operations.ak_from_categorical.from_categorical(categorical)
    assert not ak.operations.ak_is_categorical.is_categorical(not_categorical)
    assert ak.operations.ak_categories.categories(categorical).to_list() == [
        "one",
        "two",
        "three",
    ]


def test_to_categorical_masked():
    content = ak.Array(
        [
            "one",
            "two",
            "three",
            "one",
            "one",
            "two",
            "three",
            "two",
            "one",
            "two",
            "three",
            "three",
        ]
    ).layout
    mask = ak.index.Index8(
        np.array(
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
            ]
        )
    )
    array = ak.Array(ak.contents.ByteMaskedArray(mask, content, valid_when=False))
    assert not ak.operations.ak_is_categorical.is_categorical(array)
    categorical = ak.str.to_categorical(array)
    assert ak.operations.ak_is_categorical.is_categorical(categorical)
    assert ak.to_list(array) == categorical.to_list()
    assert ak.to_list(categorical.layout.content) == ["one", "two", "three"]
    not_categorical = ak.operations.ak_from_categorical.from_categorical(categorical)
    assert not ak.operations.ak_is_categorical.is_categorical(not_categorical)
    assert ak.operations.ak_categories.categories(categorical).to_list() == [
        "one",
        "two",
        "three",
    ]


def test_to_categorical_masked_again():
    content = ak.Array(
        ["one", "two", "three", "one", "one", "two", "three", "two"]
    ).layout
    index = ak.index.Index64(
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int64)
    )
    indexedarray = ak.contents.IndexedArray(index, content)
    mask = ak.index.Index8(
        np.array(
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
            ]
        )
    )
    array = ak.Array(
        ak.contents.ByteMaskedArray.simplified(mask, indexedarray, valid_when=False)
    )
    assert not ak.operations.ak_is_categorical.is_categorical(array)
    categorical = ak.str.to_categorical(array)
    assert ak.operations.ak_is_categorical.is_categorical(categorical)
    assert ak.to_list(array) == categorical.to_list()
    assert ak.to_list(categorical.layout.content) == ["one", "two", "three"]
    not_categorical = ak.operations.ak_from_categorical.from_categorical(categorical)
    assert not ak.operations.ak_is_categorical.is_categorical(not_categorical)
    assert ak.operations.ak_categories.categories(categorical).to_list() == [
        "one",
        "two",
        "three",
    ]
