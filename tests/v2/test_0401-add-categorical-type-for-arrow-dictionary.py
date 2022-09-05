# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test_same_categories():
    categories = ak._v2.Array(["one", "two", "three"])
    index1 = ak._v2.index.Index64(np.array([0, 2, 2, 1, 2, 0, 1, 0], dtype=np.int64))
    index2 = ak._v2.index.Index64(np.array([1, 1, 2, 1, 0, 0, 0, 1], dtype=np.int64))
    categorical1 = ak._v2.contents.IndexedArray(
        index1, categories.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak._v2.contents.IndexedArray(
        index2, categories.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak._v2.Array(categorical1)
    array2 = ak._v2.Array(categorical2)
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
    categories1 = ak._v2.Array(["one", "two", "three"])
    categories2 = ak._v2.Array(["three", "two", "one"])
    index1 = ak._v2.index.Index64(np.array([0, 2, 2, 1, 2, 0, 1, 0], dtype=np.int64))
    index2 = ak._v2.index.Index64(np.array([1, 1, 2, 1, 0, 0, 0, 1], dtype=np.int64))
    categorical1 = ak._v2.contents.IndexedArray(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak._v2.contents.IndexedArray(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak._v2.Array(categorical1)
    array2 = ak._v2.Array(categorical2)
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
    categories1 = ak._v2.Array(["one", "two", "three", "four"])
    categories2 = ak._v2.Array(["three", "two", "one"])
    index1 = ak._v2.index.Index64(np.array([0, 3, 0, 1, 2, 3, 1, 0], dtype=np.int64))
    index2 = ak._v2.index.Index64(np.array([1, 1, 2, 1, 0, 0, 0, 1], dtype=np.int64))
    categorical1 = ak._v2.contents.IndexedArray(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak._v2.contents.IndexedArray(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak._v2.Array(categorical1)
    array2 = ak._v2.Array(categorical2)
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
    categories1 = ak._v2.Array(["one", "two", "three"])
    categories2 = ak._v2.Array(["four", "three", "two", "one"])
    index1 = ak._v2.index.Index64(np.array([0, 2, 2, 1, 2, 0, 1, 0], dtype=np.int64))
    index2 = ak._v2.index.Index64(np.array([1, 3, 1, 1, 3, 3, 2, 1], dtype=np.int64))
    categorical1 = ak._v2.contents.IndexedArray(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak._v2.contents.IndexedArray(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak._v2.Array(categorical1)
    array2 = ak._v2.Array(categorical2)
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
    categories = ak._v2.Array(["one", "two", "three"])
    index1 = ak._v2.index.Index64(np.array([0, 2, 2, 1, -1, 0, 1, 0], dtype=np.int64))
    index2 = ak._v2.index.Index64(np.array([1, 1, 2, 1, -1, -1, 0, 1], dtype=np.int64))
    categorical1 = ak._v2.contents.IndexedOptionArray(
        index1, categories.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak._v2.contents.IndexedOptionArray(
        index2, categories.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak._v2.Array(categorical1)
    array2 = ak._v2.Array(categorical2)
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
    categories1 = ak._v2.Array(["one", "two", "three"])
    categories2 = ak._v2.Array(["three", "two", "one"])
    index1 = ak._v2.index.Index64(np.array([0, 2, 2, 1, -1, -1, 1, 0], dtype=np.int64))
    index2 = ak._v2.index.Index64(np.array([1, 1, 2, 1, 0, -1, 0, 1], dtype=np.int64))
    categorical1 = ak._v2.contents.IndexedOptionArray(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak._v2.contents.IndexedOptionArray(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak._v2.Array(categorical1)
    array2 = ak._v2.Array(categorical2)
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
    categories1 = ak._v2.Array(["one", "two", "three", "four"])
    categories2 = ak._v2.Array(["three", "two", "one"])
    index1 = ak._v2.index.Index64(np.array([0, -1, 0, 1, 2, 3, 1, 0], dtype=np.int64))
    index2 = ak._v2.index.Index64(np.array([1, -1, -1, 1, 0, 0, 0, 1], dtype=np.int64))
    categorical1 = ak._v2.contents.IndexedOptionArray(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak._v2.contents.IndexedOptionArray(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak._v2.Array(categorical1)
    array2 = ak._v2.Array(categorical2)
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
    categories1 = ak._v2.Array(["one", "two", "three"])
    categories2 = ak._v2.Array(["four", "three", "two", "one"])
    index1 = ak._v2.index.Index64(np.array([0, -1, -1, 1, 2, 0, 1, 0], dtype=np.int64))
    index2 = ak._v2.index.Index64(np.array([1, -1, 1, 1, 3, 3, 2, 1], dtype=np.int64))
    categorical1 = ak._v2.contents.IndexedOptionArray(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    categorical2 = ak._v2.contents.IndexedOptionArray(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    array1 = ak._v2.Array(categorical1)
    array2 = ak._v2.Array(categorical2)
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
    array = ak._v2.Array([1.1, 2.2, 3.3, 1.1, 2.2, 3.3, 1.1, 2.2, 3.3])
    assert not ak._v2.operations.ak_is_categorical.is_categorical(array)
    categorical = ak._v2.operations.ak_to_categorical.to_categorical(array)
    assert ak._v2.operations.ak_is_categorical.is_categorical(categorical)
    assert to_list(array) == categorical.tolist()
    assert to_list(categorical.layout.content) == [1.1, 2.2, 3.3]
    not_categorical = ak._v2.operations.ak_from_categorical.from_categorical(
        categorical
    )
    assert not ak._v2.operations.ak_is_categorical.is_categorical(not_categorical)
    assert ak._v2.operations.ak_categories.categories(categorical).tolist() == [
        1.1,
        2.2,
        3.3,
    ]


def test_to_categorical_nested():
    array = ak._v2.Array([["one", "two", "three"], [], ["one", "two"], ["three"]])
    assert not ak._v2.operations.ak_is_categorical.is_categorical(array)
    categorical = ak._v2.operations.ak_to_categorical.to_categorical(array)
    assert ak._v2.operations.ak_is_categorical.is_categorical(categorical)
    assert to_list(array) == categorical.tolist()
    not_categorical = ak._v2.operations.ak_from_categorical.from_categorical(
        categorical
    )
    assert not ak._v2.operations.ak_is_categorical.is_categorical(not_categorical)
    assert ak._v2.operations.ak_categories.categories(categorical).tolist() == [
        "one",
        "two",
        "three",
    ]


def test_to_categorical():
    array = ak._v2.Array(
        ["one", "two", "three", "one", "two", "three", "one", "two", "three"]
    )
    assert not ak._v2.operations.ak_is_categorical.is_categorical(array)
    categorical = ak._v2.operations.ak_to_categorical.to_categorical(array)
    assert ak._v2.operations.ak_is_categorical.is_categorical(categorical)
    assert to_list(array) == categorical.tolist()
    assert to_list(categorical.layout.content) == ["one", "two", "three"]
    not_categorical = ak._v2.operations.ak_from_categorical.from_categorical(
        categorical
    )
    assert not ak._v2.operations.ak_is_categorical.is_categorical(not_categorical)
    assert ak._v2.operations.ak_categories.categories(categorical).tolist() == [
        "one",
        "two",
        "three",
    ]


def test_to_categorical_none():
    array = ak._v2.Array(
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
    assert not ak._v2.operations.ak_is_categorical.is_categorical(array)
    categorical = ak._v2.operations.ak_to_categorical.to_categorical(array)
    assert ak._v2.operations.ak_is_categorical.is_categorical(categorical)
    assert to_list(array) == categorical.tolist()
    assert to_list(categorical.layout.content) == ["one", "two", "three"]
    not_categorical = ak._v2.operations.ak_from_categorical.from_categorical(
        categorical
    )
    assert not ak._v2.operations.ak_is_categorical.is_categorical(not_categorical)
    assert ak._v2.operations.ak_categories.categories(categorical).tolist() == [
        "one",
        "two",
        "three",
    ]


def test_to_categorical_masked():
    content = ak._v2.Array(
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
    mask = ak._v2.index.Index8(
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
    array = ak._v2.Array(
        ak._v2.contents.ByteMaskedArray(mask, content, valid_when=False)
    )
    assert not ak._v2.operations.ak_is_categorical.is_categorical(array)
    categorical = ak._v2.operations.ak_to_categorical.to_categorical(array)
    assert ak._v2.operations.ak_is_categorical.is_categorical(categorical)
    assert to_list(array) == categorical.tolist()
    assert to_list(categorical.layout.content) == ["one", "two", "three"]
    not_categorical = ak._v2.operations.ak_from_categorical.from_categorical(
        categorical
    )
    assert not ak._v2.operations.ak_is_categorical.is_categorical(not_categorical)
    assert ak._v2.operations.ak_categories.categories(categorical).tolist() == [
        "one",
        "two",
        "three",
    ]


def test_to_categorical_masked_again():
    content = ak._v2.Array(
        ["one", "two", "three", "one", "one", "two", "three", "two"]
    ).layout
    index = ak._v2.index.Index64(
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int64)
    )
    indexedarray = ak._v2.contents.IndexedArray(index, content)
    mask = ak._v2.index.Index8(
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
    array = ak._v2.Array(
        ak._v2.contents.ByteMaskedArray(mask, indexedarray, valid_when=False)
    )
    assert not ak._v2.operations.ak_is_categorical.is_categorical(array)
    categorical = ak._v2.operations.ak_to_categorical.to_categorical(array)
    assert ak._v2.operations.ak_is_categorical.is_categorical(categorical)
    assert to_list(array) == categorical.tolist()
    assert to_list(categorical.layout.content) == ["one", "two", "three"]
    not_categorical = ak._v2.operations.ak_from_categorical.from_categorical(
        categorical
    )
    assert not ak._v2.operations.ak_is_categorical.is_categorical(not_categorical)
    assert ak._v2.operations.ak_categories.categories(categorical).tolist() == [
        "one",
        "two",
        "three",
    ]


@pytest.mark.skip(reason="Fix issues for categorical type")
def test_typestr():
    assert (
        str(
            ak._v2.operations.type(
                ak._v2.operations.ak_to_categorical.to_categorical(
                    ak._v2.Array([1.1, 2.2, 2.2, 3.3])
                )
            )
        )
        == "4 * categorical[type=float64]"
    )
    assert (
        str(
            ak._v2.operations.type(
                ak._v2.operations.ak_to_categorical.to_categorical(
                    ak._v2.Array([1.1, 2.2, None, 2.2, 3.3])
                )
            )
        )
        == "5 * categorical[type=?float64]"
    )
    assert (
        str(
            ak._v2.operations.type(
                ak._v2.operations.ak_to_categorical.to_categorical(
                    ak._v2.Array(["one", "two", "two", "three"])
                )
            )
        )
        == "4 * categorical[type=string]"
    )
    assert (
        str(
            ak._v2.operations.type(
                ak._v2.operations.ak_to_categorical.to_categorical(
                    ak._v2.Array(["one", "two", None, "two", "three"])
                )
            )
        )
        == "5 * categorical[type=?string]"
    )


def test_zip():
    x = ak._v2.Array([1.1, 2.2, 3.3])
    y = ak._v2.Array(["one", "two", "three"])
    assert ak.zip({"x": x, "y": y}).tolist() == [
        {"x": 1.1, "y": "one"},
        {"x": 2.2, "y": "two"},
        {"x": 3.3, "y": "three"},
    ]
    y = ak._v2.operations.ak_to_categorical.to_categorical(y)
    assert ak.zip({"x": x, "y": y}).tolist() == [
        {"x": 1.1, "y": "one"},
        {"x": 2.2, "y": "two"},
        {"x": 3.3, "y": "three"},
    ]


pyarrow = pytest.importorskip("pyarrow")


def test_arrow_nomask():
    array = ak._v2.Array([1.1, 2.2, 3.3, 4.4, None])
    assert (
        str(
            ak._v2.operations.type(
                ak._v2.operations.from_arrow(ak._v2.operations.to_arrow(array))
            )
        )
        == "5 * ?float64"
    )
    assert (
        str(
            ak._v2.operations.type(
                ak._v2.operations.from_arrow(ak._v2.operations.to_arrow(array[:-1]))
            )
        )
        == "4 * ?float64"
    )
