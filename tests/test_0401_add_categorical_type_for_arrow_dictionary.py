# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_same_categories():
    categories = ak.Array(["one", "two", "three"])
    index1 = ak.index.Index64(np.array([0, 2, 2, 1, 2, 0, 1, 0], dtype=np.int64))
    index2 = ak.index.Index64(np.array([1, 1, 2, 1, 0, 0, 0, 1], dtype=np.int64))
    categorical1 = ak.contents.IndexedArray(
        index1, categories.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak.contents.IndexedArray(
        index2, categories.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak.Array(categorical1)
    array2 = ak.Array(categorical2)
    assert (array1 == array2).to_list() == [
        False,
        False,
        True,
        True,
        False,
        True,
        False,
        False,
    ]


def test_different_categories():
    categories1 = ak.Array(["one", "two", "three"])
    categories2 = ak.Array(["three", "two", "one"])
    index1 = ak.index.Index64(np.array([0, 2, 2, 1, 2, 0, 1, 0], dtype=np.int64))
    index2 = ak.index.Index64(np.array([1, 1, 2, 1, 0, 0, 0, 1], dtype=np.int64))
    categorical1 = ak.contents.IndexedArray(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak.contents.IndexedArray(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak.Array(categorical1)
    array2 = ak.Array(categorical2)
    assert (array1 == array2).to_list() == [
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
    ]


def test_one_extra():
    categories1 = ak.Array(["one", "two", "three", "four"])
    categories2 = ak.Array(["three", "two", "one"])
    index1 = ak.index.Index64(np.array([0, 3, 0, 1, 2, 3, 1, 0], dtype=np.int64))
    index2 = ak.index.Index64(np.array([1, 1, 2, 1, 0, 0, 0, 1], dtype=np.int64))
    categorical1 = ak.contents.IndexedArray(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak.contents.IndexedArray(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak.Array(categorical1)
    array2 = ak.Array(categorical2)
    assert (array1 == array2).to_list() == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_two_extra():
    categories1 = ak.Array(["one", "two", "three"])
    categories2 = ak.Array(["four", "three", "two", "one"])
    index1 = ak.index.Index64(np.array([0, 2, 2, 1, 2, 0, 1, 0], dtype=np.int64))
    index2 = ak.index.Index64(np.array([1, 3, 1, 1, 3, 3, 2, 1], dtype=np.int64))
    categorical1 = ak.contents.IndexedArray(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak.contents.IndexedArray(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak.Array(categorical1)
    array2 = ak.Array(categorical2)
    assert (array1 == array2).to_list() == [
        False,
        False,
        True,
        False,
        False,
        True,
        True,
        False,
    ]


def test_option_same_categories():
    categories = ak.Array(["one", "two", "three"])
    index1 = ak.index.Index64(np.array([0, 2, 2, 1, -1, 0, 1, 0], dtype=np.int64))
    index2 = ak.index.Index64(np.array([1, 1, 2, 1, -1, -1, 0, 1], dtype=np.int64))
    categorical1 = ak.contents.IndexedOptionArray(
        index1, categories.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak.contents.IndexedOptionArray(
        index2, categories.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak.Array(categorical1)
    array2 = ak.Array(categorical2)
    assert (array1 == array2).to_list() == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_option_different_categories():
    categories1 = ak.Array(["one", "two", "three"])
    categories2 = ak.Array(["three", "two", "one"])
    index1 = ak.index.Index64(np.array([0, 2, 2, 1, -1, -1, 1, 0], dtype=np.int64))
    index2 = ak.index.Index64(np.array([1, 1, 2, 1, 0, -1, 0, 1], dtype=np.int64))
    categorical1 = ak.contents.IndexedOptionArray(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak.contents.IndexedOptionArray(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak.Array(categorical1)
    array2 = ak.Array(categorical2)
    assert (array1 == array2).to_list() == [
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
    ]


def test_option_one_extra():
    categories1 = ak.Array(["one", "two", "three", "four"])
    categories2 = ak.Array(["three", "two", "one"])
    index1 = ak.index.Index64(np.array([0, -1, 0, 1, 2, 3, 1, 0], dtype=np.int64))
    index2 = ak.index.Index64(np.array([1, -1, -1, 1, 0, 0, 0, 1], dtype=np.int64))
    categorical1 = ak.contents.IndexedOptionArray(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak.contents.IndexedOptionArray(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak.Array(categorical1)
    array2 = ak.Array(categorical2)
    assert (array1 == array2).to_list() == [
        False,
        True,
        False,
        True,
        True,
        False,
        False,
        False,
    ]


def test_option_two_extra():
    categories1 = ak.Array(["one", "two", "three"])
    categories2 = ak.Array(["four", "three", "two", "one"])
    index1 = ak.index.Index64(np.array([0, -1, -1, 1, 2, 0, 1, 0], dtype=np.int64))
    index2 = ak.index.Index64(np.array([1, -1, 1, 1, 3, 3, 2, 1], dtype=np.int64))
    categorical1 = ak.contents.IndexedOptionArray(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak.contents.IndexedOptionArray(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak.Array(categorical1)
    array2 = ak.Array(categorical2)
    assert (array1 == array2).to_list() == [
        False,
        True,
        False,
        False,
        False,
        True,
        True,
        False,
    ]


pyarrow = pytest.importorskip("pyarrow")


def test_arrow_nomask():
    array = ak.Array([1.1, 2.2, 3.3, 4.4, None])
    assert (
        str(ak.operations.type(ak.operations.from_arrow(ak.operations.to_arrow(array))))
        == "5 * ?float64"
    )
    assert (
        str(
            ak.operations.type(
                ak.operations.from_arrow(ak.operations.to_arrow(array[:-1]))
            )
        )
        == "4 * ?float64"
    )
