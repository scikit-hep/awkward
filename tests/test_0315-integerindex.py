# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_integerindex_null():
    a = ak.Array([[0, 1, 2], None, [5, 6], [7]])
    b = ak.Array([[0, 1, 2], [3, 4], [5, 6], [7]])
    c = ak.Array([[1], [1], [0], [0]])
    d = ak.Array([[1], None, [0], [0]])
    e = ak.Array([[1], None, None, [0]])

    assert ak.to_list(a[c]) == [[1], None, [5], [7]]
    assert ak.to_list(a[d]) == [[1], None, [5], [7]]
    assert ak.to_list(a[e]) == [[1], None, None, [7]]
    assert ak.to_list(b[c]) == [[1], [4], [5], [7]]
    assert ak.to_list(b[d]) == [[1], None, [5], [7]]
    assert ak.to_list(b[e]) == [[1], None, None, [7]]


def test_boolindex_null():
    a = ak.Array([[0, 1, 2], None, [5, 6]])
    b = ak.Array([[0, 1, 2], [3, 4], [5, 6]])
    c = ak.Array([[False, True, False], [False, True], [True, False]])
    d = ak.Array([[False, True, False], None, [True, False]])
    e = ak.Array([[False, True, False], None, None])

    assert ak.to_list(a[c]) == [[1], None, [5]]
    assert ak.to_list(a[d]) == [[1], None, [5]]
    assert ak.to_list(a[e]) == [[1], None, None]
    assert ak.to_list(b[c]) == [[1], [4], [5]]
    assert ak.to_list(b[d]) == [[1], None, [5]]
    assert ak.to_list(b[e]) == [[1], None, None]

    b2 = b.mask[[True, False, True]]
    assert ak.to_list(b2[c]) == [[1], None, [5]]
    assert ak.to_list(b2[d]) == [[1], None, [5]]
    assert ak.to_list(b2[e]) == [[1], None, None]


def test_integerindex_null_more():
    f = ak.Array([[0, None, 2], None, [3, 4], []])
    g1 = ak.Array([[1, 2, None], None, [], [None]])
    g2 = ak.Array([[], None, None, []])
    g3 = ak.Array([[], [], [], []])

    assert ak.to_list(f[g1]) == [[None, 2, None], None, [], [None]]
    assert ak.to_list(f[g2]) == [[], None, None, []]
    assert ak.to_list(f[g3]) == [[], None, [], []]

    a = ak.Array([[0, 1, 2, None], None])
    b = ak.Array([[2, 1, None, 3], None])
    assert ak.to_list(a[b]) == [[2, 1, None, None], None]
    b = ak.Array([[2, 1, None, 3], []])
    assert ak.to_list(a[b]) == [[2, 1, None, None], None]
    b = ak.Array([[2, 1, None, 3], [0, 1]])
    assert ak.to_list(a[b]) == [[2, 1, None, None], None]

    a = ak.Array([[[0, 1, 2, None], None], [[3, 4], [5]], None, [[6]]])
    b = ak.Array([[[2, 1, None, 3], [0, 1]], [[0], None], None, [None]])
    c = ak.Array(
        [
            [[False, True, None, False], [False, True]],
            [[True, False], None],
            None,
            [None],
        ]
    )
    assert ak.to_list(a[b]) == [[[2, 1, None, None], None], [[3], None], None, [None]]
    assert ak.to_list(a[c]) == [[[1, None], None], [[4], None], None, [None]]


def test_silly_stuff():
    a = ak.Array([[0, 1, 2], 3])
    b = [[2], [0]]
    with pytest.raises(ValueError):
        a[b]

    a = ak.Array([[0, 1, 2], [3, 4], [5, 6], [7]])
    b = ak.Array([[0, 2], None])
    assert ak.to_list(a[b]) == [[0, 2], None]
    b = ak.Array([[0, 2], None, None, None, None, None])
    with pytest.raises(ValueError):
        a[b]
