# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_same_categories():
    categories = ak.Array(["one", "two", "three"])
    index1 = ak.layout.Index64(np.array([0, 2, 2, 1, 2, 0, 1, 0], dtype=np.int64))
    index2 = ak.layout.Index64(np.array([1, 1, 2, 1, 0, 0, 0, 1], dtype=np.int64))
    categorical1 = ak.layout.IndexedArray64(
        index1, categories.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak.layout.IndexedArray64(
        index2, categories.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak.Array(categorical1)
    array2 = ak.Array(categorical2)
    assert (array1 == array2).tolist() == [
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
    index1 = ak.layout.Index64(np.array([0, 2, 2, 1, 2, 0, 1, 0], dtype=np.int64))
    index2 = ak.layout.Index64(np.array([1, 1, 2, 1, 0, 0, 0, 1], dtype=np.int64))
    categorical1 = ak.layout.IndexedArray64(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak.layout.IndexedArray64(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak.Array(categorical1)
    array2 = ak.Array(categorical2)
    assert (array1 == array2).tolist() == [
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
    index1 = ak.layout.Index64(np.array([0, 3, 0, 1, 2, 3, 1, 0], dtype=np.int64))
    index2 = ak.layout.Index64(np.array([1, 1, 2, 1, 0, 0, 0, 1], dtype=np.int64))
    categorical1 = ak.layout.IndexedArray64(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak.layout.IndexedArray64(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak.Array(categorical1)
    array2 = ak.Array(categorical2)
    assert (array1 == array2).tolist() == [
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
    index1 = ak.layout.Index64(np.array([0, 2, 2, 1, 2, 0, 1, 0], dtype=np.int64))
    index2 = ak.layout.Index64(np.array([1, 3, 1, 1, 3, 3, 2, 1], dtype=np.int64))
    categorical1 = ak.layout.IndexedArray64(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak.layout.IndexedArray64(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak.Array(categorical1)
    array2 = ak.Array(categorical2)
    assert (array1 == array2).tolist() == [
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
    index1 = ak.layout.Index64(np.array([0, 2, 2, 1, -1, 0, 1, 0], dtype=np.int64))
    index2 = ak.layout.Index64(np.array([1, 1, 2, 1, -1, -1, 0, 1], dtype=np.int64))
    categorical1 = ak.layout.IndexedOptionArray64(
        index1, categories.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak.layout.IndexedOptionArray64(
        index2, categories.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak.Array(categorical1)
    array2 = ak.Array(categorical2)
    assert (array1 == array2).tolist() == [
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
    index1 = ak.layout.Index64(np.array([0, 2, 2, 1, -1, -1, 1, 0], dtype=np.int64))
    index2 = ak.layout.Index64(np.array([1, 1, 2, 1, 0, -1, 0, 1], dtype=np.int64))
    categorical1 = ak.layout.IndexedOptionArray64(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak.layout.IndexedOptionArray64(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak.Array(categorical1)
    array2 = ak.Array(categorical2)
    assert (array1 == array2).tolist() == [
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
    index1 = ak.layout.Index64(np.array([0, -1, 0, 1, 2, 3, 1, 0], dtype=np.int64))
    index2 = ak.layout.Index64(np.array([1, -1, -1, 1, 0, 0, 0, 1], dtype=np.int64))
    categorical1 = ak.layout.IndexedOptionArray64(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak.layout.IndexedOptionArray64(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak.Array(categorical1)
    array2 = ak.Array(categorical2)
    assert (array1 == array2).tolist() == [
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
    index1 = ak.layout.Index64(np.array([0, -1, -1, 1, 2, 0, 1, 0], dtype=np.int64))
    index2 = ak.layout.Index64(np.array([1, -1, 1, 1, 3, 3, 2, 1], dtype=np.int64))
    categorical1 = ak.layout.IndexedOptionArray64(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak.layout.IndexedOptionArray64(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak.Array(categorical1)
    array2 = ak.Array(categorical2)
    assert (array1 == array2).tolist() == [
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
    assert not ak.is_categorical(array)
    categorical = ak.to_categorical(array)
    assert ak.is_categorical(categorical)
    assert ak.to_list(array) == categorical.tolist()
    assert ak.to_list(categorical.layout.content) == [1.1, 2.2, 3.3]
    not_categorical = ak.from_categorical(categorical)
    assert not ak.is_categorical(not_categorical)
    assert ak.categories(categorical).tolist() == [1.1, 2.2, 3.3]


def test_to_categorical_nested():
    array = ak.Array([["one", "two", "three"], [], ["one", "two"], ["three"]])
    assert not ak.is_categorical(array)
    categorical = ak.to_categorical(array)
    assert ak.is_categorical(categorical)
    assert ak.to_list(array) == categorical.tolist()
    not_categorical = ak.from_categorical(categorical)
    assert not ak.is_categorical(not_categorical)
    assert ak.categories(categorical).tolist() == ["one", "two", "three"]


def test_to_categorical():
    array = ak.Array(
        ["one", "two", "three", "one", "two", "three", "one", "two", "three"]
    )
    assert not ak.is_categorical(array)
    categorical = ak.to_categorical(array)
    assert ak.is_categorical(categorical)
    assert ak.to_list(array) == categorical.tolist()
    assert ak.to_list(categorical.layout.content) == ["one", "two", "three"]
    not_categorical = ak.from_categorical(categorical)
    assert not ak.is_categorical(not_categorical)
    assert ak.categories(categorical).tolist() == ["one", "two", "three"]


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
    assert not ak.is_categorical(array)
    categorical = ak.to_categorical(array)
    assert ak.is_categorical(categorical)
    assert ak.to_list(array) == categorical.tolist()
    assert ak.to_list(categorical.layout.content) == ["one", "two", "three"]
    not_categorical = ak.from_categorical(categorical)
    assert not ak.is_categorical(not_categorical)
    assert ak.categories(categorical).tolist() == ["one", "two", "three"]


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
    mask = ak.layout.Index8(
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
    array = ak.Array(ak.layout.ByteMaskedArray(mask, content, valid_when=False))
    assert not ak.is_categorical(array)
    categorical = ak.to_categorical(array)
    assert ak.is_categorical(categorical)
    assert ak.to_list(array) == categorical.tolist()
    assert ak.to_list(categorical.layout.content) == ["one", "two", "three"]
    not_categorical = ak.from_categorical(categorical)
    assert not ak.is_categorical(not_categorical)
    assert ak.categories(categorical).tolist() == ["one", "two", "three"]


def test_to_categorical_masked_again():
    content = ak.Array(
        ["one", "two", "three", "one", "one", "two", "three", "two"]
    ).layout
    index = ak.layout.Index64(
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int64)
    )
    indexedarray = ak.layout.IndexedArray64(index, content)
    mask = ak.layout.Index8(
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
    array = ak.Array(ak.layout.ByteMaskedArray(mask, indexedarray, valid_when=False))
    assert not ak.is_categorical(array)
    categorical = ak.to_categorical(array)
    assert ak.is_categorical(categorical)
    assert ak.to_list(array) == categorical.tolist()
    assert ak.to_list(categorical.layout.content) == ["one", "two", "three"]
    not_categorical = ak.from_categorical(categorical)
    assert not ak.is_categorical(not_categorical)
    assert ak.categories(categorical).tolist() == ["one", "two", "three"]


def test_typestr():
    if not ak._util.py27:
        assert (
            str(ak.type(ak.to_categorical(ak.Array([1.1, 2.2, 2.2, 3.3]))))
            == "4 * categorical[type=float64]"
        )
        assert (
            str(ak.type(ak.to_categorical(ak.Array([1.1, 2.2, None, 2.2, 3.3]))))
            == "5 * categorical[type=?float64]"
        )
        assert (
            str(ak.type(ak.to_categorical(ak.Array(["one", "two", "two", "three"]))))
            == "4 * categorical[type=string]"
        )
        assert (
            str(
                ak.type(
                    ak.to_categorical(ak.Array(["one", "two", None, "two", "three"]))
                )
            )
            == "5 * categorical[type=option[string]]"
        )


def test_zip():
    x = ak.Array([1.1, 2.2, 3.3])
    y = ak.Array(["one", "two", "three"])
    assert ak.zip({"x": x, "y": y}).tolist() == [
        {"x": 1.1, "y": "one"},
        {"x": 2.2, "y": "two"},
        {"x": 3.3, "y": "three"},
    ]
    y = ak.to_categorical(y)
    assert ak.zip({"x": x, "y": y}).tolist() == [
        {"x": 1.1, "y": "one"},
        {"x": 2.2, "y": "two"},
        {"x": 3.3, "y": "three"},
    ]


pyarrow = pytest.importorskip("pyarrow")


def test_arrow_nomask():
    array = ak.Array([1.1, 2.2, 3.3, 4.4, None])
    assert str(ak.type(ak.from_arrow(ak.to_arrow(array)))) == "5 * ?float64"
    assert str(ak.type(ak.from_arrow(ak.to_arrow(array[:-1])))) == "4 * ?float64"
