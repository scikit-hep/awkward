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


def test_to_categorical():
    array = awkward1.Array(["one", "two", "three", "one", "two", "three", "one", "two", "three"])
    categorical = awkward1.to_categorical(array)
    assert awkward1.to_list(array) == categorical.tolist()
    assert awkward1.to_list(categorical.layout.content) == ["one", "two", "three"]


def test_to_categorical_none():
    array = awkward1.Array(["one", "two", "three", None, "one", "two", "three", None, "one", "two", "three", None])
    categorical = awkward1.to_categorical(array)
    assert awkward1.to_list(array) == categorical.tolist()
    assert awkward1.to_list(categorical.layout.content) == ["one", "two", "three"]

def test_to_categorical_masked():
    content = awkward1.Array(["one", "two", "three", "one", "one", "two", "three", "two", "one", "two", "three", "three"]).layout
    mask = awkward1.layout.Index8(numpy.array([False, False, False, True, False, False, False, True, False, False, False, True]))
    array = awkward1.Array(awkward1.layout.ByteMaskedArray(mask, content, valid_when=False))
    categorical = awkward1.to_categorical(array)
    assert awkward1.to_list(array) == categorical.tolist()
    assert awkward1.to_list(categorical.layout.content) == ["one", "two", "three"]

def test_to_categorical_masked():
    content = awkward1.Array(["one", "two", "three", "one", "one", "two", "three", "two"]).layout
    index = awkward1.layout.Index64(numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray64(index, content)
    mask = awkward1.layout.Index8(numpy.array([False, False, False, True, False, False, False, True, False, False, False, True]))
    array = awkward1.Array(awkward1.layout.ByteMaskedArray(mask, indexedarray, valid_when=False))
    categorical = awkward1.to_categorical(array)
    assert awkward1.to_list(array) == categorical.tolist()
    assert awkward1.to_list(categorical.layout.content) == ["one", "two", "three"]
