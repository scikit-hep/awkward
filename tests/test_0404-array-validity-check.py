# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward

def test_NumpyArray():
    array = awkward.layout.NumpyArray(numpy.array(["1chchc", "1chchc", "2sss", "3", "4", "5"], dtype=object), parameters={"__array__": "categorical"})
    assert awkward.is_valid(awkward.Array(array)) == False
    # FIXME? assert array.is_unique() == False
    array2 = awkward.layout.NumpyArray(numpy.array([5, 6, 1, 3, 4, 5]))
    assert array2.is_unique() == False

def test_ListOffsetArray():
    array = awkward.from_iter(["one", "two", "three", "four", "five"], highlevel=False)
    assert awkward.to_list(array.sort(0, True, True)) == ["five", "four", "one", "three", "two"]
    assert array.is_unique() == True

    array2 = awkward.from_iter(["one", "two", "one", "four", "two"], highlevel=False)
    assert awkward.to_list(array2.sort(0, True, True)) == ["four", "one", "one", "two", "two"]
    assert array2.is_unique() == False

def test_same_categories():
    categories = awkward.Array(["one", "two", "three"])
    index1 = awkward.layout.Index64(numpy.array([0, 2, 2, 1, 2, 0, 1, 0], dtype=numpy.int64))
    index2 = awkward.layout.Index64(numpy.array([1, 1, 2, 1, 0, 0, 0, 1], dtype=numpy.int64))
    categorical1 = awkward.layout.IndexedArray64(index1, categories.layout, parameters={"__array__": "categorical"})
    categorical2 = awkward.layout.IndexedArray64(index2, categories.layout, parameters={"__array__": "categorical"})
    array1 = awkward.Array(categorical1)
    assert awkward.to_list(categorical1.sort(0, True, True)) == ['one', 'one', 'one', 'three', 'three', 'three', 'two', 'two']
    #assert categorical1.is_unique() == False
    array2 = awkward.Array(categorical2)
    assert array1.tolist() == ['one', 'three', 'three', 'two', 'three', 'one', 'two', 'one']
    assert array2.tolist() == ['two', 'two', 'three', 'two', 'one', 'one', 'one', 'two']

    assert (array1 == array2).tolist() == [False, False, True, True, False, True, False, False]
