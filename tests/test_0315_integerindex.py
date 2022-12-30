# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_integerindex_null():
    a = ak.highlevel.Array([[0, 1, 2], None, [5, 6], [7]]).layout
    b = ak.highlevel.Array([[0, 1, 2], [3, 4], [5, 6], [7]]).layout
    c = ak.highlevel.Array([[1], [1], [0], [0]]).layout
    d = ak.highlevel.Array([[1], None, [0], [0]]).layout
    e = ak.highlevel.Array([[1], None, None, [0]]).layout

    assert to_list(a[c]) == [[1], None, [5], [7]]
    assert to_list(a[d]) == [[1], None, [5], [7]]
    assert to_list(a[e]) == [[1], None, None, [7]]
    assert to_list(b[c]) == [[1], [4], [5], [7]]
    assert to_list(b[d]) == [[1], None, [5], [7]]
    assert to_list(b[e]) == [[1], None, None, [7]]

    assert a.to_typetracer()[c].form == a[c].form
    assert a.to_typetracer()[d].form == a[d].form
    assert a.to_typetracer()[e].form == a[e].form
    assert b.to_typetracer()[c].form == b[c].form
    assert b.to_typetracer()[d].form == b[d].form
    assert b.to_typetracer()[e].form == b[e].form


def test_boolindex_null():
    a = ak.highlevel.Array([[0, 1, 2], None, [5, 6]]).layout
    b = ak.highlevel.Array([[0, 1, 2], [3, 4], [5, 6]]).layout
    c = ak.highlevel.Array([[False, True, False], [False, True], [True, False]]).layout
    d = ak.highlevel.Array([[False, True, False], None, [True, False]]).layout
    e = ak.highlevel.Array([[False, True, False], None, None]).layout

    assert to_list(a[c]) == [[1], None, [5]]
    assert to_list(a[d]) == [[1], None, [5]]
    assert to_list(a[e]) == [[1], None, None]
    assert to_list(b[c]) == [[1], [4], [5]]
    assert to_list(b[d]) == [[1], None, [5]]
    assert to_list(b[e]) == [[1], None, None]

    assert a.to_typetracer()[c].form == a[c].form
    assert a.to_typetracer()[d].form == a[d].form
    assert a.to_typetracer()[e].form == a[e].form
    assert b.to_typetracer()[c].form == b[c].form
    assert b.to_typetracer()[d].form == b[d].form
    assert b.to_typetracer()[e].form == b[e].form

    b2 = ak.contents.ByteMaskedArray(
        ak.index.Index8([True, False, True]),
        b,
        valid_when=True,
    )

    assert to_list(b2[c]) == [[1], None, [5]]
    assert to_list(b2[d]) == [[1], None, [5]]
    assert to_list(b2[e]) == [[1], None, None]

    assert b2.to_typetracer()[c].form == b2[c].form
    assert b2.to_typetracer()[d].form == b2[d].form
    assert b2.to_typetracer()[e].form == b2[e].form


def test_integerindex_null_more():
    f = ak.highlevel.Array([[0, None, 2], None, [3, 4], []]).layout
    g1 = ak.highlevel.Array([[1, 2, None], None, [], [None]]).layout
    g2 = ak.highlevel.Array([[], None, None, []]).layout
    g3 = ak.highlevel.Array([[], [], [], []]).layout

    assert to_list(f[g1]) == [[None, 2, None], None, [], [None]]
    assert to_list(f[g2]) == [[], None, None, []]
    assert to_list(f[g3]) == [[], None, [], []]
    assert f.to_typetracer()[g1].form == f[g1].form
    assert f.to_typetracer()[g2].form == f[g2].form
    assert f.to_typetracer()[g3].form == f[g3].form

    a = ak.highlevel.Array([[0, 1, 2, None], None]).layout
    b = ak.highlevel.Array([[2, 1, None, 3], None]).layout

    assert to_list(a[b]) == [[2, 1, None, None], None]
    assert a.to_typetracer()[b].form == a[b].form

    b = ak.highlevel.Array([[2, 1, None, 3], []]).layout

    assert to_list(a[b]) == [[2, 1, None, None], None]
    assert a.to_typetracer()[b].form == a[b].form

    b = ak.highlevel.Array([[2, 1, None, 3], [0, 1]]).layout
    assert to_list(a[b]) == [[2, 1, None, None], None]
    assert a.to_typetracer()[b].form == a[b].form


def test_integerindex_null_more_2():
    a = ak.highlevel.Array([[[0, 1, 2, None], None], [[3, 4], [5]], None, [[6]]]).layout
    b = ak.highlevel.Array(
        [[[2, 1, None, 3], [0, 1]], [[0], None], None, [None]]
    ).layout
    c = ak.highlevel.Array(
        [
            [[False, True, None, False], [False, True]],
            [[True, False], None],
            None,
            [None],
        ]
    ).layout

    assert to_list(a[b]) == [
        [[2, 1, None, None], None],
        [[3], None],
        None,
        [None],
    ]
    assert to_list(a[c]) == [[[1, None], None], [[4], None], None, [None]]
    assert a.to_typetracer()[c].form == a[c].form


def test_silly_stuff():
    a = ak.highlevel.Array([[0, 1, 2], 3]).layout
    b = [[2], [0]]

    with pytest.raises(IndexError):
        a[b]

    a = ak.highlevel.Array([[0, 1, 2], [3, 4], [5, 6], [7]]).layout

    b = ak.highlevel.Array([[0, 2], None, [1], None]).layout

    assert to_list(a[b]) == [[0, 2], None, [6], None]

    b = ak.highlevel.Array([[0, 2], None, None, None, None, None]).layout

    with pytest.raises(IndexError):
        a[b]
