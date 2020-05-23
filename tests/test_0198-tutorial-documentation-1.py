# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_singletons():
    array = awkward1.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])
    assert awkward1.to_list(awkward1.singletons(array)) == [[1.1], [2.2], [], [3.3], [], [], [4.4], [5.5]]

    assert awkward1.to_list(awkward1.singletons(awkward1.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5]))) == [[1.1], [2.2], [], [3.3], [], [], [4.4], [5.5]]
    assert awkward1.to_list(awkward1.singletons(awkward1.Array([[1.1, 2.2, None], [3.3, None], [None], [4.4, 5.5]]))) == [[[1.1], [2.2], []], [[3.3], []], [[]], [[4.4], [5.5]]]
    assert awkward1.to_list(awkward1.singletons(awkward1.Array([[[1.1, 2.2, None]], [[3.3, None]], [[None]], [[4.4, 5.5]]]))) == [[[[1.1], [2.2], []]], [[[3.3], []]], [[[]]], [[[4.4], [5.5]]]]

def test_firsts():
    array = awkward1.singletons(awkward1.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5]))
    assert awkward1.to_list(awkward1.firsts(awkward1.singletons(awkward1.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])), axis=1)) == [1.1, 2.2, None, 3.3, None, None, 4.4, 5.5]
    assert awkward1.to_list(awkward1.firsts(awkward1.singletons(awkward1.Array([[1.1, 2.2, None], [3.3, None], [None], [4.4, 5.5]])), axis=2)) == [[1.1, 2.2, None], [3.3, None], [None], [4.4, 5.5]]
    assert awkward1.to_list(awkward1.firsts(awkward1.singletons(awkward1.Array([[[1.1, 2.2, None]], [[3.3, None]], [[None]], [[4.4, 5.5]]])), axis=3)) == [[[1.1, 2.2, None]], [[3.3, None]], [[None]], [[4.4, 5.5]]]

def test_allow_missing():
    array = awkward1.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])
    awkward1.to_numpy(array)
    with pytest.raises(ValueError):
        awkward1.to_numpy(array, allow_missing=False)

def test_flatten0():
    array = awkward1.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])
    assert awkward1.to_list(awkward1.flatten(array, axis=0)) == [1.1, 2.2, 3.3, 4.4, 5.5]

    content0 = awkward1.from_iter([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5], highlevel=False)
    content1 = awkward1.from_iter(["one", None, "two", None, "three"], highlevel=False)
    array = awkward1.Array(awkward1.layout.UnionArray8_64(
                awkward1.layout.Index8(numpy.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0], dtype=numpy.int8)),
                awkward1.layout.Index64(numpy.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 4, 7], dtype=numpy.int64)),
                [content0, content1]))
    assert awkward1.to_list(array) == [1.1, "one", 2.2, None, None, "two", 3.3, None, None, None, 4.4, "three", 5.5]
    assert awkward1.to_list(awkward1.flatten(array, axis=0)) == [1.1, "one", 2.2, "two", 3.3, 4.4, "three", 5.5]
