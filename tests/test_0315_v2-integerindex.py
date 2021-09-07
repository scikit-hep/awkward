# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

fixed_value_error = False

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_integerindex_null():
    a = ak.Array([[0, 1, 2], None, [5, 6], [7]])
    b = ak.Array([[0, 1, 2], [3, 4], [5, 6], [7]])
    c = ak.Array([[1], [1], [0], [0]])
    d = ak.Array([[1], None, [0], [0]])
    e = ak.Array([[1], None, None, [0]])

    a = v1_to_v2(a.layout)
    b = v1_to_v2(b.layout)
    c = v1_to_v2(c.layout)
    d = v1_to_v2(d.layout)
    e = v1_to_v2(e.layout)

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
    b2 = b.mask[[True, False, True]]

    a = v1_to_v2(a.layout)
    b = v1_to_v2(b.layout)
    c = v1_to_v2(c.layout)
    d = v1_to_v2(d.layout)
    e = v1_to_v2(e.layout)

    assert ak.to_list(a[c]) == [[1], None, [5]]
    assert ak.to_list(a[d]) == [[1], None, [5]]
    assert ak.to_list(a[e]) == [[1], None, None]
    assert ak.to_list(b[c]) == [[1], [4], [5]]
    assert ak.to_list(b[d]) == [[1], None, [5]]
    assert ak.to_list(b[e]) == [[1], None, None]

    b2 = v1_to_v2(b2.layout)

    assert ak.to_list(b2[c]) == [[1], None, [5]]
    assert ak.to_list(b2[d]) == [[1], None, [5]]
    assert ak.to_list(b2[e]) == [[1], None, None]


def test_integerindex_null_more():
    f = ak.Array([[0, None, 2], None, [3, 4], []])
    g1 = ak.Array([[1, 2, None], None, [], [None]])
    g2 = ak.Array([[], None, None, []])
    g3 = ak.Array([[], [], [], []])

    f = v1_to_v2(f.layout)
    g1 = v1_to_v2(g1.layout)
    g2 = v1_to_v2(g2.layout)
    g3 = v1_to_v2(g3.layout)

    assert ak.to_list(f[g1]) == [[None, 2, None], None, [], [None]]
    assert ak.to_list(f[g2]) == [[], None, None, []]
    assert ak.to_list(f[g3]) == [[], None, [], []]

    a = ak.Array([[0, 1, 2, None], None])
    b = ak.Array([[2, 1, None, 3], None])

    a = v1_to_v2(a.layout)
    b = v1_to_v2(b.layout)

    assert ak.to_list(a[b]) == [[2, 1, None, None], None]

    b = ak.Array([[2, 1, None, 3], []])
    b = v1_to_v2(b.layout)

    assert ak.to_list(a[b]) == [[2, 1, None, None], None]

    b = ak.Array([[2, 1, None, 3], [0, 1]])
    b = v1_to_v2(b.layout)
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

    a = v1_to_v2(a.layout)
    b = v1_to_v2(b.layout)
    c = v1_to_v2(c.layout)

    if fixed_value_error:
        assert ak.to_list(a[b]) == [
            [[2, 1, None, None], None],
            [[3], None],
            None,
            [None],
        ]
        assert ak.to_list(a[c]) == [[[1, None], None], [[4], None], None, [None]]


def test_silly_stuff():
    a = ak.Array([[0, 1, 2], 3])
    b = [[2], [0]]

    a = v1_to_v2(a.layout)

    with pytest.raises(IndexError):
        a[b]

    a = ak.Array([[0, 1, 2], [3, 4], [5, 6], [7]])
    a = v1_to_v2(a.layout)

    b = ak.Array([[0, 2], None, [1], None])
    b = v1_to_v2(b.layout)

    assert ak.to_list(a[b]) == [[0, 2], None, [6], None]

    b = ak.Array([[0, 2], None, None, None, None, None])
    b = v1_to_v2(b.layout)

    with pytest.raises(IndexError):
        a[b]
