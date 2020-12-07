# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward

def test_NumpyArray():
    array = awkward.layout.NumpyArray(numpy.array(["1chchc", "1chchc", "2sss", "3", "4", "5"], dtype=object), parameters={"__array__": "categorical"})
    assert awkward.is_valid(awkward.Array(array)) == False

def test_same_categories():
    categories = awkward.Array(["one", "two", "three"])
    index1 = awkward.layout.Index64(numpy.array([0, 2, 2, 1, 2, 0, 1, 0], dtype=numpy.int64))
    index2 = awkward.layout.Index64(numpy.array([1, 1, 2, 1, 0, 0, 0, 1], dtype=numpy.int64))
    categorical1 = awkward.layout.IndexedArray64(index1, categories.layout, parameters={"__array__": "categorical"})
    categorical2 = awkward.layout.IndexedArray64(index2, categories.layout, parameters={"__array__": "categorical"})
    array1 = awkward.Array(categorical1)
    array2 = awkward.Array(categorical2)
    assert (array1 == array2).tolist() == [False, False, True, True, False, True, False, False]
