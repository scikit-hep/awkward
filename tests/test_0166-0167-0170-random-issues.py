# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_0166_IndexedOptionArray():
    array = awkward1.Array([[2, 3, 5], None, [], [7, 11], None, [13], None, [17, 19]])
    assert awkward1.to_list(awkward1.prod(array, axis=-1)) == [30, None, 1, 77, None, 13, None, 323]

    array = awkward1.Array([[[2, 3], [5]], None, [], [[7], [11]], None, [[13]], None, [[17, 19]]])
    assert awkward1.to_list(awkward1.prod(array, axis=-1)) == [[6, 5], None, [], [7, 11], None, [13], None, [323]]

    array = awkward1.Array([[[2, 3], None, [5]], [], [[7], [11]], [[13]], [None, [17], [19]]])
    awkward1.to_list(awkward1.prod(array, axis=-1)) == [[6, None, 5], [], [7, 11], [13], [None, 17, 19]]

    array = awkward1.Array([[6, None, 5], [], [7, 11], [13], [None, 17, 19]])
    assert awkward1.to_list(awkward1.prod(array, axis=-1)) == [30, 1, 77, 13, 323]

def test_0166_ByteMaskedArray():
    content = awkward1.from_iter([[2, 3, 5], [999], [], [7, 11], [], [13], [123, 999], [17, 19]], highlevel=False)
    mask = awkward1.layout.Index8(numpy.array([0, 1, 0, 0, 1, 0, 1, 0], dtype=numpy.int8))
    array = awkward1.Array(awkward1.layout.ByteMaskedArray(mask, content, valid_when=False))
    assert awkward1.to_list(array) == [[2, 3, 5], None, [], [7, 11], None, [13], None, [17, 19]]
    assert awkward1.to_list(awkward1.prod(array, axis=-1)) == [30, None, 1, 77, None, 13, None, 323]

    content = awkward1.from_iter([[[2, 3], [5]], [[999]], [], [[7], [11]], [], [[13]], [[123], [999]], [[17, 19]]], highlevel=False)
    mask = awkward1.layout.Index8(numpy.array([0, 1, 0, 0, 1, 0, 1, 0], dtype=numpy.int8))
    array = awkward1.Array(awkward1.layout.ByteMaskedArray(mask, content, valid_when=False))
    assert awkward1.to_list(array) == [[[2, 3], [5]], None, [], [[7], [11]], None, [[13]], None, [[17, 19]]]
    assert awkward1.to_list(awkward1.prod(array, axis=-1)) == [[6, 5], None, [], [7, 11], None, [13], None, [323]]

    content = awkward1.from_iter([[2, 3], [999], [5], [7], [11], [13], [], [17], [19]], highlevel=False)
    mask = awkward1.layout.Index8(numpy.array([0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=numpy.int8))
    bytemasked = awkward1.layout.ByteMaskedArray(mask, content, valid_when=False)
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9], dtype=numpy.int64))
    array = awkward1.Array(awkward1.layout.ListOffsetArray64(offsets, bytemasked))
    array = awkward1.Array([[[2, 3], None, [5]], [], [[7], [11]], [[13]], [None, [17], [19]]])
    assert awkward1.to_list(awkward1.prod(array, axis=-1)) == [[6, None, 5], [], [7, 11], [13], [None, 17, 19]]

    content = awkward1.from_iter([6, None, 5, 7, 11, 13, None, 17, 19], highlevel=False)
    mask = awkward1.layout.Index8(numpy.array([0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=numpy.int8))
    bytemasked = awkward1.layout.ByteMaskedArray(mask, content, valid_when=False)
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9], dtype=numpy.int64))
    array = awkward1.Array(awkward1.layout.ListOffsetArray64(offsets, bytemasked))
    assert awkward1.to_list(array) == [[6, None, 5], [], [7, 11], [13], [None, 17, 19]]
    assert awkward1.to_list(awkward1.prod(array, axis=-1)) == [30, 1, 77, 13, 323]

def test_0167_strings():
    array = awkward1.Array(["one", "two", "three", "two", "two", "one", "three"])
    assert awkward1.to_list(array == "two") == [False, True, False, True, True, False, False]
    assert awkward1.to_list("two" == array) == [False, True, False, True, True, False, False]
    assert awkward1.to_list(array == ["two"]) == [False, True, False, True, True, False, False]
    assert awkward1.to_list(["two"] == array) == [False, True, False, True, True, False, False]
    assert awkward1.to_list(array == awkward1.Array(["two"])) == [False, True, False, True, True, False, False]
    assert awkward1.to_list(awkward1.Array(["two"]) == array) == [False, True, False, True, True, False, False]
    assert awkward1.to_list(array < array) == [[False, False, False], [False, False, False], [False, False, False, False, False], [False, False, False], [False, False, False], [False, False, False], [False, False, False, False, False]]
    assert awkward1.to_list(array <= array) == [[True, True, True], [True, True, True], [True, True, True, True, True], [True, True, True], [True, True, True], [True, True, True], [True, True, True, True, True]]

    array = awkward1.Array([["one", "two", "three"], [], ["two"], ["two", "one"], ["three"]])
    assert awkward1.to_list(array == "two") == [[False, True, False], [], [True], [True, False], [False]]
    assert awkward1.to_list("two" == array) == [[False, True, False], [], [True], [True, False], [False]]
    assert awkward1.to_list(array == ["two"]) == [[False, True, False], [], [True], [True, False], [False]]
    assert awkward1.to_list(["two"] == array) == [[False, True, False], [], [True], [True, False], [False]]
    assert awkward1.to_list(array == awkward1.Array(["two"])) == [[False, True, False], [], [True], [True, False], [False]]
    assert awkward1.to_list(awkward1.Array(["two"]) == array) == [[False, True, False], [], [True], [True, False], [False]]
    assert awkward1.to_list(array < array) == [[[False, False, False], [False, False, False], [False, False, False, False, False]], [], [[False, False, False]], [[False, False, False], [False, False, False]], [[False, False, False, False, False]]]
    assert awkward1.to_list(array <= array) == [[[True, True, True], [True, True, True], [True, True, True, True, True]], [], [[True, True, True]], [[True, True, True], [True, True, True]], [[True, True, True, True, True]]]

    array = awkward1.Array([["one", "two", "three"], [], ["two"], ["two", "one"], ["three"]])
    assert awkward1.to_list(array == ["three", "two", "one", "one", "three"]) == [[False, False, True], [], [False], [False, True], [True]]
    assert awkward1.to_list(["three", "two", "one", "one", "three"] == array) == [[False, False, True], [], [False], [False, True], [True]]
    assert awkward1.to_list(array == awkward1.Array(["three", "two", "one", "one", "three"])) == [[False, False, True], [], [False], [False, True], [True]]
    assert awkward1.to_list(awkward1.Array(["three", "two", "one", "one", "three"]) == array) == [[False, False, True], [], [False], [False, True], [True]]

def test_0167_bytestrings():
    array = awkward1.Array([b"one", b"two", b"three", b"two", b"two", b"one", b"three"])
    assert awkward1.to_list(array == b"two") == [False, True, False, True, True, False, False]
    assert awkward1.to_list(b"two" == array) == [False, True, False, True, True, False, False]
    assert awkward1.to_list(array == [b"two"]) == [False, True, False, True, True, False, False]
    assert awkward1.to_list([b"two"] == array) == [False, True, False, True, True, False, False]
    assert awkward1.to_list(array == awkward1.Array([b"two"])) == [False, True, False, True, True, False, False]
    assert awkward1.to_list(awkward1.Array([b"two"]) == array) == [False, True, False, True, True, False, False]
    assert awkward1.to_list(array < array) == [[False, False, False], [False, False, False], [False, False, False, False, False], [False, False, False], [False, False, False], [False, False, False], [False, False, False, False, False]]
    assert awkward1.to_list(array <= array) == [[True, True, True], [True, True, True], [True, True, True, True, True], [True, True, True], [True, True, True], [True, True, True], [True, True, True, True, True]]

    array = awkward1.Array([[b"one", b"two", b"three"], [], [b"two"], [b"two", b"one"], [b"three"]])
    assert awkward1.to_list(array == b"two") == [[False, True, False], [], [True], [True, False], [False]]
    assert awkward1.to_list(b"two" == array) == [[False, True, False], [], [True], [True, False], [False]]
    assert awkward1.to_list(array == [b"two"]) == [[False, True, False], [], [True], [True, False], [False]]
    assert awkward1.to_list([b"two"] == array) == [[False, True, False], [], [True], [True, False], [False]]
    assert awkward1.to_list(array == awkward1.Array([b"two"])) == [[False, True, False], [], [True], [True, False], [False]]
    assert awkward1.to_list(awkward1.Array([b"two"]) == array) == [[False, True, False], [], [True], [True, False], [False]]
    assert awkward1.to_list(array < array) == [[[False, False, False], [False, False, False], [False, False, False, False, False]], [], [[False, False, False]], [[False, False, False], [False, False, False]], [[False, False, False, False, False]]]
    assert awkward1.to_list(array <= array) == [[[True, True, True], [True, True, True], [True, True, True, True, True]], [], [[True, True, True]], [[True, True, True], [True, True, True]], [[True, True, True, True, True]]]

    array = awkward1.Array([[b"one", b"two", b"three"], [], [b"two"], [b"two", b"one"], [b"three"]])
    assert awkward1.to_list(array == [b"three", b"two", b"one", b"one", b"three"]) == [[False, False, True], [], [False], [False, True], [True]]
    assert awkward1.to_list([b"three", b"two", b"one", b"one", b"three"] == array) == [[False, False, True], [], [False], [False, True], [True]]
    assert awkward1.to_list(array == awkward1.Array([b"three", b"two", b"one", b"one", b"three"])) == [[False, False, True], [], [False], [False, True], [True]]
    assert awkward1.to_list(awkward1.Array([b"three", b"two", b"one", b"one", b"three"]) == array) == [[False, False, True], [], [False], [False, True], [True]]
