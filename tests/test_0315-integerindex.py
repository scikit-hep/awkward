# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import awkward1

def test_integerindex_null():
    a = awkward1.Array([[0, 1, 2], None, [5, 6], [7]])
    b = awkward1.Array([[0, 1, 2], [3, 4], [5, 6], [7]])
    c = awkward1.Array([[1], [1], [0], [0]])
    d = awkward1.Array([[1], None, [0], [0]])
    e = awkward1.Array([[1], None, None, [0]])

    assert awkward1.to_list(a[c]) == [[1], None, [5], [7]]
    assert awkward1.to_list(a[d]) == [[1], None, [5], [7]]
    assert awkward1.to_list(a[e]) == [[1], None, None, [7]]
    assert awkward1.to_list(b[c]) == [[1], [4], [5], [7]]
    assert awkward1.to_list(b[d]) == [[1], None, [5], [7]]
    assert awkward1.to_list(b[e]) == [[1], None, None, [7]]

def test_boolindex_null():
    a = awkward1.Array([[0, 1, 2], None, [5, 6]])
    b = awkward1.Array([[0, 1, 2], [3, 4], [5, 6]])
    c = awkward1.Array([[False, True, False], [False, True], [True, False]])
    d = awkward1.Array([[False, True, False], None, [True, False]])
    e = awkward1.Array([[False, True, False], None, None])

    assert awkward1.to_list(a[c]) == [[1], None, [5]]
    assert awkward1.to_list(a[d]) == [[1], None, [5]]
    assert awkward1.to_list(a[e]) == [[1], None, None]
    assert awkward1.to_list(b[c]) == [[1], [4], [5]]
    assert awkward1.to_list(b[d]) == [[1], None, [5]]
    assert awkward1.to_list(b[e]) == [[1], None, None]

    b2 = b.mask[[True, False, True]]
    assert awkward1.to_list(b2[c]) == [[1], None, [5]]
    assert awkward1.to_list(b2[d]) == [[1], None, [5]]
    assert awkward1.to_list(b2[e]) == [[1], None, None]

def test_integerindex_null_more():
    f = awkward1.Array([[0, None, 2], None, [3, 4], []])
    g1 = awkward1.Array([[1, 2, None], None, [], [None]])
    g2 = awkward1.Array([[], None, None, []])
    g3 = awkward1.Array([[], [], [], []])

    assert awkward1.to_list(f[g1]) == [[None, 2, None], None, [], [None]]
    assert awkward1.to_list(f[g2]) == [[], None, None, []]
    assert awkward1.to_list(f[g3]) == [[], None, [], []]

    a = awkward1.Array([[0, 1, 2, None], None])
    b = awkward1.Array([[2, 1, None, 3], None])
    assert awkward1.to_list(a[b]) == [[2, 1, None, None], None]
    b = awkward1.Array([[2, 1, None, 3], []])
    assert awkward1.to_list(a[b]) == [[2, 1, None, None], None]
    b = awkward1.Array([[2, 1, None, 3], [0, 1]])
    assert awkward1.to_list(a[b]) == [[2, 1, None, None], None]

    a = awkward1.Array([[[0, 1, 2, None], None], [[3, 4], [5]], None, [[6]]])
    b = awkward1.Array([[[2, 1, None, 3], [0, 1]], [[0], None], None, [None]])
    c = awkward1.Array([[[False, True, None, False], [False, True]], [[True, False], None], None, [None]])
    assert awkward1.to_list(a[b]) == [[[2, 1, None, None], None], [[3], None], None, [None]]
    assert awkward1.to_list(a[c]) == [[[1, None], None], [[4], None], None, [None]]

def test_silly_stuff():
    a = awkward1.Array([[0, 1, 2], 3])
    b = [[2], [0]]
    with pytest.raises(ValueError):
        a[b]

    a = awkward1.Array([[0, 1, 2], [3, 4], [5, 6], [7]])
    b = awkward1.Array([[0, 2], None])
    assert awkward1.to_list(a[b]) == [[0, 2], None]
    b = awkward1.Array([[0, 2], None, None, None, None, None])
    with pytest.raises(ValueError):
        a[b]

