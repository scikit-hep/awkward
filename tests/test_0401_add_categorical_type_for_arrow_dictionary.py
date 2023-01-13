# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

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


def test_to_categorical_numbers():
    array = ak.Array([1.1, 2.2, 3.3, 1.1, 2.2, 3.3, 1.1, 2.2, 3.3])
    assert not ak.operations.ak_is_categorical.is_categorical(array)
    categorical = ak.operations.ak_to_categorical.to_categorical(array)
    assert ak.operations.ak_is_categorical.is_categorical(categorical)
    assert to_list(array) == categorical.to_list()
    assert to_list(categorical.layout.content) == [1.1, 2.2, 3.3]
    not_categorical = ak.operations.ak_from_categorical.from_categorical(categorical)
    assert not ak.operations.ak_is_categorical.is_categorical(not_categorical)
    assert ak.operations.ak_categories.categories(categorical).to_list() == [
        1.1,
        2.2,
        3.3,
    ]


def test_to_categorical_nested():
    array = ak.Array([["one", "two", "three"], [], ["one", "two"], ["three"]])
    assert not ak.operations.ak_is_categorical.is_categorical(array)
    categorical = ak.operations.ak_to_categorical.to_categorical(array)
    assert ak.operations.ak_is_categorical.is_categorical(categorical)
    assert to_list(array) == categorical.to_list()
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
    categorical = ak.operations.ak_to_categorical.to_categorical(array)
    assert ak.operations.ak_is_categorical.is_categorical(categorical)
    assert to_list(array) == categorical.to_list()
    assert to_list(categorical.layout.content) == ["one", "two", "three"]
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
    categorical = ak.operations.ak_to_categorical.to_categorical(array)
    assert ak.operations.ak_is_categorical.is_categorical(categorical)
    assert to_list(array) == categorical.to_list()
    assert to_list(categorical.layout.content) == ["one", "two", "three"]
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
    categorical = ak.operations.ak_to_categorical.to_categorical(array)
    assert ak.operations.ak_is_categorical.is_categorical(categorical)
    assert to_list(array) == categorical.to_list()
    assert to_list(categorical.layout.content) == ["one", "two", "three"]
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
    categorical = ak.operations.ak_to_categorical.to_categorical(array)
    assert ak.operations.ak_is_categorical.is_categorical(categorical)
    assert to_list(array) == categorical.to_list()
    assert to_list(categorical.layout.content) == ["one", "two", "three"]
    not_categorical = ak.operations.ak_from_categorical.from_categorical(categorical)
    assert not ak.operations.ak_is_categorical.is_categorical(not_categorical)
    assert ak.operations.ak_categories.categories(categorical).to_list() == [
        "one",
        "two",
        "three",
    ]


@pytest.mark.skip(reason="Fix issues for categorical type")
def test_typestr():
    assert (
        str(
            ak.operations.type(
                ak.operations.ak_to_categorical.to_categorical(
                    ak.Array([1.1, 2.2, 2.2, 3.3])
                )
            )
        )
        == "4 * categorical[type=float64]"
    )
    assert (
        str(
            ak.operations.type(
                ak.operations.ak_to_categorical.to_categorical(
                    ak.Array([1.1, 2.2, None, 2.2, 3.3])
                )
            )
        )
        == "5 * categorical[type=?float64]"
    )
    assert (
        str(
            ak.operations.type(
                ak.operations.ak_to_categorical.to_categorical(
                    ak.Array(["one", "two", "two", "three"])
                )
            )
        )
        == "4 * categorical[type=string]"
    )
    assert (
        str(
            ak.operations.type(
                ak.operations.ak_to_categorical.to_categorical(
                    ak.Array(["one", "two", None, "two", "three"])
                )
            )
        )
        == "5 * categorical[type=?string]"
    )


def test_zip():
    x = ak.Array([1.1, 2.2, 3.3])
    y = ak.Array(["one", "two", "three"])
    assert ak.zip({"x": x, "y": y}).to_list() == [
        {"x": 1.1, "y": "one"},
        {"x": 2.2, "y": "two"},
        {"x": 3.3, "y": "three"},
    ]
    y = ak.operations.ak_to_categorical.to_categorical(y)
    assert ak.zip({"x": x, "y": y}).to_list() == [
        {"x": 1.1, "y": "one"},
        {"x": 2.2, "y": "two"},
        {"x": 3.3, "y": "three"},
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
