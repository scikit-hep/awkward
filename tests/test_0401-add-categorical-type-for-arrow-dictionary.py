# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test_same_categories():
    categories = awkward1.Array(["one", "two", "three"])
    index1 = awkward1.layout.Index64(numpy.array([0, 2, 2, 1, 2, 0, 1, 0], dtype=numpy.int64))
    index2 = awkward1.layout.Index64(numpy.array([1, 1, 2, 1, 0, 0, 0, 1], dtype=numpy.int64))
    categorical1 = awkward1.layout.IndexedArray64(index1, categories.layout, parameters={"__array__": "categorical"})
    categorical2 = awkward1.layout.IndexedArray64(index2, categories.layout, parameters={"__array__": "categorical"})
    array1 = awkward1.Array(categorical1)
    array2 = awkward1.Array(categorical2)
    assert (array1 == array2).tolist() == [False, False, True, True, False, True, False, False]


def test_different_categories():
    categories1 = awkward1.Array(["one", "two", "three"])
    categories2 = awkward1.Array(["three", "two", "one"])
    index1 = awkward1.layout.Index64(numpy.array([0, 2, 2, 1, 2, 0, 1, 0], dtype=numpy.int64))
    index2 = awkward1.layout.Index64(numpy.array([1, 1, 2, 1, 0, 0, 0, 1], dtype=numpy.int64))
    categorical1 = awkward1.layout.IndexedArray64(index1, categories1.layout, parameters={"__array__": "categorical"})
    categorical2 = awkward1.layout.IndexedArray64(index2, categories2.layout, parameters={"__array__": "categorical"})
    array1 = awkward1.Array(categorical1)
    array2 = awkward1.Array(categorical2)
    assert (array1 == array2).tolist() == [False, False, False, True, True, False, False, False]


def test_one_extra():
    categories1 = awkward1.Array(["one", "two", "three", "four"])
    categories2 = awkward1.Array(["three", "two", "one"])
    index1 = awkward1.layout.Index64(numpy.array([0, 3, 0, 1, 2, 3, 1, 0], dtype=numpy.int64))
    index2 = awkward1.layout.Index64(numpy.array([1, 1, 2, 1, 0, 0, 0, 1], dtype=numpy.int64))
    categorical1 = awkward1.layout.IndexedArray64(index1, categories1.layout, parameters={"__array__": "categorical"})
    categorical2 = awkward1.layout.IndexedArray64(index2, categories2.layout, parameters={"__array__": "categorical"})
    array1 = awkward1.Array(categorical1)
    array2 = awkward1.Array(categorical2)
    assert (array1 == array2).tolist() == [False, False, True, True, True, False, False, False]


def test_two_extra():
    categories1 = awkward1.Array(["one", "two", "three"])
    categories2 = awkward1.Array(["four", "three", "two", "one"])
    index1 = awkward1.layout.Index64(numpy.array([0, 2, 2, 1, 2, 0, 1, 0], dtype=numpy.int64))
    index2 = awkward1.layout.Index64(numpy.array([1, 3, 1, 1, 3, 3, 2, 1], dtype=numpy.int64))
    categorical1 = awkward1.layout.IndexedArray64(index1, categories1.layout, parameters={"__array__": "categorical"})
    categorical2 = awkward1.layout.IndexedArray64(index2, categories2.layout, parameters={"__array__": "categorical"})
    array1 = awkward1.Array(categorical1)
    array2 = awkward1.Array(categorical2)
    assert (array1 == array2).tolist() == [False, False, True, False, False, True, True, False]


def test_option_same_categories():
    categories = awkward1.Array(["one", "two", "three"])
    index1 = awkward1.layout.Index64(numpy.array([0, 2, 2, 1, -1,  0, 1, 0], dtype=numpy.int64))
    index2 = awkward1.layout.Index64(numpy.array([1, 1, 2, 1, -1, -1, 0, 1], dtype=numpy.int64))
    categorical1 = awkward1.layout.IndexedOptionArray64(index1, categories.layout, parameters={"__array__": "categorical"})
    categorical2 = awkward1.layout.IndexedOptionArray64(index2, categories.layout, parameters={"__array__": "categorical"})
    array1 = awkward1.Array(categorical1)
    array2 = awkward1.Array(categorical2)
    assert (array1 == array2).tolist() == [False, False, True, True, True, False, False, False]


def test_option_different_categories():
    categories1 = awkward1.Array(["one", "two", "three"])
    categories2 = awkward1.Array(["three", "two", "one"])
    index1 = awkward1.layout.Index64(numpy.array([0, 2, 2, 1, -1, -1, 1, 0], dtype=numpy.int64))
    index2 = awkward1.layout.Index64(numpy.array([1, 1, 2, 1,  0, -1, 0, 1], dtype=numpy.int64))
    categorical1 = awkward1.layout.IndexedOptionArray64(index1, categories1.layout, parameters={"__array__": "categorical"})
    categorical2 = awkward1.layout.IndexedOptionArray64(index2, categories2.layout, parameters={"__array__": "categorical"})
    array1 = awkward1.Array(categorical1)
    array2 = awkward1.Array(categorical2)
    assert (array1 == array2).tolist() == [False, False, False, True, False, True, False, False]


def test_option_one_extra():
    categories1 = awkward1.Array(["one", "two", "three", "four"])
    categories2 = awkward1.Array(["three", "two", "one"])
    index1 = awkward1.layout.Index64(numpy.array([0, -1,  0, 1, 2, 3, 1, 0], dtype=numpy.int64))
    index2 = awkward1.layout.Index64(numpy.array([1, -1, -1, 1, 0, 0, 0, 1], dtype=numpy.int64))
    categorical1 = awkward1.layout.IndexedOptionArray64(index1, categories1.layout, parameters={"__array__": "categorical"})
    categorical2 = awkward1.layout.IndexedOptionArray64(index2, categories2.layout, parameters={"__array__": "categorical"})
    array1 = awkward1.Array(categorical1)
    array2 = awkward1.Array(categorical2)
    assert (array1 == array2).tolist() == [False, True, False, True, True, False, False, False]


def test_option_two_extra():
    categories1 = awkward1.Array(["one", "two", "three"])
    categories2 = awkward1.Array(["four", "three", "two", "one"])
    index1 = awkward1.layout.Index64(numpy.array([0, -1, -1, 1, 2, 0, 1, 0], dtype=numpy.int64))
    index2 = awkward1.layout.Index64(numpy.array([1, -1,  1, 1, 3, 3, 2, 1], dtype=numpy.int64))
    categorical1 = awkward1.layout.IndexedOptionArray64(index1, categories1.layout, parameters={"__array__": "categorical"})
    categorical2 = awkward1.layout.IndexedOptionArray64(index2, categories2.layout, parameters={"__array__": "categorical"})
    array1 = awkward1.Array(categorical1)
    array2 = awkward1.Array(categorical2)
    assert (array1 == array2).tolist() == [False, True, False, False, False, True, True, False]


def test_to_categorical_numbers():
    array = awkward1.Array([1.1, 2.2, 3.3, 1.1, 2.2, 3.3, 1.1, 2.2, 3.3])
    assert not awkward1.is_categorical(array)
    categorical = awkward1.to_categorical(array)
    assert awkward1.is_categorical(categorical)
    assert awkward1.to_list(array) == categorical.tolist()
    assert awkward1.to_list(categorical.layout.content) == [1.1, 2.2, 3.3]
    not_categorical = awkward1.from_categorical(categorical)
    assert not awkward1.is_categorical(not_categorical)
    assert awkward1.categories(categorical).tolist() == [1.1, 2.2, 3.3]


def test_to_categorical_nested():
    array = awkward1.Array([["one", "two", "three"], [], ["one", "two"], ["three"]])
    assert not awkward1.is_categorical(array)
    categorical = awkward1.to_categorical(array)
    assert awkward1.is_categorical(categorical)
    assert awkward1.to_list(array) == categorical.tolist()
    not_categorical = awkward1.from_categorical(categorical)
    assert not awkward1.is_categorical(not_categorical)
    assert awkward1.categories(categorical).tolist() == ["one", "two", "three"]


def test_to_categorical():
    array = awkward1.Array(["one", "two", "three", "one", "two", "three", "one", "two", "three"])
    assert not awkward1.is_categorical(array)
    categorical = awkward1.to_categorical(array)
    assert awkward1.is_categorical(categorical)
    assert awkward1.to_list(array) == categorical.tolist()
    assert awkward1.to_list(categorical.layout.content) == ["one", "two", "three"]
    not_categorical = awkward1.from_categorical(categorical)
    assert not awkward1.is_categorical(not_categorical)
    assert awkward1.categories(categorical).tolist() == ["one", "two", "three"]


def test_to_categorical_none():
    array = awkward1.Array(["one", "two", "three", None, "one", "two", "three", None, "one", "two", "three", None])
    assert not awkward1.is_categorical(array)
    categorical = awkward1.to_categorical(array)
    assert awkward1.is_categorical(categorical)
    assert awkward1.to_list(array) == categorical.tolist()
    assert awkward1.to_list(categorical.layout.content) == ["one", "two", "three"]
    not_categorical = awkward1.from_categorical(categorical)
    assert not awkward1.is_categorical(not_categorical)
    assert awkward1.categories(categorical).tolist() == ["one", "two", "three"]


def test_to_categorical_masked():
    content = awkward1.Array(["one", "two", "three", "one", "one", "two", "three", "two", "one", "two", "three", "three"]).layout
    mask = awkward1.layout.Index8(numpy.array([False, False, False, True, False, False, False, True, False, False, False, True]))
    array = awkward1.Array(awkward1.layout.ByteMaskedArray(mask, content, valid_when=False))
    assert not awkward1.is_categorical(array)
    categorical = awkward1.to_categorical(array)
    assert awkward1.is_categorical(categorical)
    assert awkward1.to_list(array) == categorical.tolist()
    assert awkward1.to_list(categorical.layout.content) == ["one", "two", "three"]
    not_categorical = awkward1.from_categorical(categorical)
    assert not awkward1.is_categorical(not_categorical)
    assert awkward1.categories(categorical).tolist() == ["one", "two", "three"]


def test_to_categorical_masked():
    content = awkward1.Array(["one", "two", "three", "one", "one", "two", "three", "two"]).layout
    index = awkward1.layout.Index64(numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray64(index, content)
    mask = awkward1.layout.Index8(numpy.array([False, False, False, True, False, False, False, True, False, False, False, True]))
    array = awkward1.Array(awkward1.layout.ByteMaskedArray(mask, indexedarray, valid_when=False))
    assert not awkward1.is_categorical(array)
    categorical = awkward1.to_categorical(array)
    assert awkward1.is_categorical(categorical)
    assert awkward1.to_list(array) == categorical.tolist()
    assert awkward1.to_list(categorical.layout.content) == ["one", "two", "three"]
    not_categorical = awkward1.from_categorical(categorical)
    assert not awkward1.is_categorical(not_categorical)
    assert awkward1.categories(categorical).tolist() == ["one", "two", "three"]


def test_typestr():
    if not awkward1._util.py27:
        assert str(awkward1.type(awkward1.to_categorical(awkward1.Array([1.1, 2.2, 2.2, 3.3])))) == "4 * categorical[type=float64]"
        assert str(awkward1.type(awkward1.to_categorical(awkward1.Array([1.1, 2.2, None, 2.2, 3.3])))) == "5 * categorical[type=?float64]"
        assert str(awkward1.type(awkward1.to_categorical(awkward1.Array(["one", "two", "two", "three"])))) == "4 * categorical[type=string]"
        assert str(awkward1.type(awkward1.to_categorical(awkward1.Array(["one", "two", None, "two", "three"])))) == "5 * categorical[type=option[string]]"


def test_zip():
    x = awkward1.Array([1.1, 2.2, 3.3])
    y = awkward1.Array(["one", "two", "three"])
    assert awkward1.zip({"x": x, "y": y}).tolist() == [{"x": 1.1, "y": "one"}, {"x": 2.2, "y": "two"}, {"x": 3.3, "y": "three"}]
    y = awkward1.to_categorical(y)
    assert awkward1.zip({"x": x, "y": y}).tolist() == [{"x": 1.1, "y": "one"}, {"x": 2.2, "y": "two"}, {"x": 3.3, "y": "three"}]


pyarrow = pytest.importorskip("pyarrow")


def test_arrow_nomask():
    array = awkward1.Array([1.1, 2.2, 3.3, 4.4, None])
    assert str(awkward1.type(awkward1.from_arrow(awkward1.to_arrow(array)))) == "5 * ?float64"
    assert str(awkward1.type(awkward1.from_arrow(awkward1.to_arrow(array[:-1])))) == "4 * ?float64"
