# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_numpyarray_getitem_bystrides():
    a = np.arange(10)
    b = ak.layout.NumpyArray(a)
    b = v1_to_v2(b)

    assert b[3] == a[3]
    assert b[-3] == a[-3]
    assert ak.to_list(b[()]) == ak.to_list(a[()])
    assert ak.to_list(b[slice(None)]) == ak.to_list(a[slice(None)])
    assert ak.to_list(b[slice(3, 7)]) == ak.to_list(a[slice(3, 7)])
    assert ak.to_list(b[slice(3, 100)]) == ak.to_list(a[slice(3, 100)])
    assert ak.to_list(b[slice(-100, 7)]) == ak.to_list(a[slice(-100, 7)])
    assert ak.to_list(b[slice(3, -3)]) == ak.to_list(a[slice(3, -3)])
    assert ak.to_list(b[slice(1, 7, 2)]) == ak.to_list(a[slice(1, 7, 2)])
    assert ak.to_list(b[slice(-8, 7, 2)]) == ak.to_list(a[slice(-8, 7, 2)])
    assert ak.to_list(b[slice(None, 7, 2)]) == ak.to_list(a[slice(None, 7, 2)])
    assert ak.to_list(b[slice(None, -3, 2)]) == ak.to_list(a[slice(None, -3, 2)])
    assert ak.to_list(b[slice(8, None, -1)]) == ak.to_list(a[slice(8, None, -1)])
    assert ak.to_list(b[slice(8, None, -2)]) == ak.to_list(a[slice(8, None, -2)])
    assert ak.to_list(b[slice(-2, None, -2)]) == ak.to_list(a[slice(-2, None, -2)])

    a = np.arange(7 * 5).reshape(7, 5)
    b = ak.layout.NumpyArray(a)
    b = v1_to_v2(b)

    assert ak.to_list(b[()]) == ak.to_list(a[()])
    assert ak.to_list(b[3]) == ak.to_list(a[3])
    assert ak.to_list(b[(3, 2)]) == ak.to_list(a[3, 2])
    assert ak.to_list(b[slice(1, 4)]) == ak.to_list(a[slice(1, 4)])
    assert ak.to_list(b[(3, slice(1, 4))]) == ak.to_list(a[3, slice(1, 4)])
    assert ak.to_list(b[(slice(1, 4), 3)]) == ak.to_list(a[slice(1, 4), 3])
    assert ak.to_list(b[(slice(1, 4), slice(2, None))]) == ak.to_list(
        a[slice(1, 4), slice(2, None)]
    )
    assert ak.to_list(b[(slice(None, None, -1), slice(2, None))]) == ak.to_list(
        a[slice(None, None, -1), slice(2, None)]
    )

    assert ak.to_list(b[:, np.newaxis, :]) == ak.to_list(a[:, np.newaxis, :])
    assert ak.to_list(b[np.newaxis, :, np.newaxis, :, np.newaxis]) == ak.to_list(
        a[np.newaxis, :, np.newaxis, :, np.newaxis]
    )

    if not ak._util.py27:
        assert ak.to_list(b[Ellipsis, 3]) == ak.to_list(a[Ellipsis, 3])
        assert ak.to_list(b[Ellipsis, 3, 2]) == ak.to_list(a[Ellipsis, 3, 2])
        assert ak.to_list(b[3, Ellipsis]) == ak.to_list(a[3, Ellipsis])
        assert ak.to_list(b[3, 2, Ellipsis]) == ak.to_list(a[3, 2, Ellipsis])


def test_numpyarray_getitem_next():
    a = np.arange(10)
    b = ak.layout.NumpyArray(a)
    c = np.array([7, 3, 3, 5])
    b = v1_to_v2(b)

    assert ak.to_list(b[c]) == ak.to_list(a[c])
    assert b.typetracer[c].form == b[c].form
    c = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    assert ak.to_list(b[c]) == ak.to_list(a[c])
    assert b.typetracer[c].form == b[c].form
    c = np.array([False, False, False, True, True, True, False, True, False, True])
    assert ak.to_list(b[c]) == ak.to_list(a[c])
    assert b.typetracer[c].form == b[c].form
    c = np.array([], dtype=int)
    assert ak.to_list(b[c]) == ak.to_list(a[c])
    assert b.typetracer[c].form == b[c].form

    a = np.arange(10 * 3).reshape(10, 3)
    b = ak.layout.NumpyArray(a)
    c = np.array([7, 3, 3, 5])
    b = v1_to_v2(b)

    assert ak.to_list(b[c]) == ak.to_list(a[c])
    assert b.typetracer[c].form == b[c].form
    c = np.array([False, False, False, True, True, True, False, True, False, True])
    assert ak.to_list(b[c]) == ak.to_list(a[c])
    assert b.typetracer[c].form == b[c].form
    c = np.array([], dtype=int)
    assert ak.to_list(b[c]) == ak.to_list(a[c])
    assert b.typetracer[c].form == b[c].form

    a = np.arange(7 * 5).reshape(7, 5)
    b = ak.layout.NumpyArray(a)
    b = v1_to_v2(b)

    c1 = np.array([], np.int64)
    c2 = np.array([], np.int64)

    assert ak.to_list(b[c1, c2]) == ak.to_list(a[c1, c2])
    assert ak.Array(b[c1, c2]).ndim == a[c1, c2].ndim
    assert b.typetracer[c1, c2].form == b[c1, c2].form

    a = np.arange(7 * 5).reshape(7, 5)
    b = ak.layout.NumpyArray(a)
    b = v1_to_v2(b)
    c1 = np.array([4, 1, 1, 3])
    c2 = np.array([2, 2, 0, 1])
    assert ak.to_list(b[c1, c2]) == ak.to_list(a[c1, c2])
    assert b.typetracer[c1, c2].form == b[c1, c2].form

    c = np.array([False, False, True, True, False, True, True])
    assert ak.to_list(b[c]) == ak.to_list(a[c])
    assert b.typetracer[c].form == b[c].form

    c = np.array([], dtype=int)
    assert ak.to_list(b[c]) == ak.to_list(a[c])
    c1 = np.array([], dtype=int)
    c2 = np.array([], dtype=int)
    assert ak.to_list(b[c1, c2]) == ak.to_list(a[c1, c2])
    assert b.typetracer[c1, c2].form == b[c1, c2].form

    a = np.arange(7 * 5).reshape(7, 5)
    b = ak.layout.NumpyArray(a)
    b = v1_to_v2(b)
    c = np.array([2, 0, 0, 1])
    assert ak.to_list(b[1:4, c]) == ak.to_list(a[1:4, c])
    assert ak.to_list(b[c, 1:4]) == ak.to_list(a[c, 1:4])
    assert ak.to_list(b[1:4, np.newaxis, c]) == ak.to_list(a[1:4, np.newaxis, c])
    assert ak.to_list(b[c, np.newaxis, 1:4]) == ak.to_list(a[c, np.newaxis, 1:4])
    assert ak.to_list(b[1:4, np.newaxis, np.newaxis, c]) == ak.to_list(
        a[1:4, np.newaxis, np.newaxis, c]
    )
    assert ak.to_list(b[c, np.newaxis, np.newaxis, 1:4]) == ak.to_list(
        a[c, np.newaxis, np.newaxis, 1:4]
    )
    if not ak._util.py27:
        assert ak.to_list(b[Ellipsis, c]) == ak.to_list(a[Ellipsis, c])
        assert ak.to_list(b[c, Ellipsis]) == ak.to_list(a[c, Ellipsis])


def test_numpyarray_getitem_next_2():
    a = np.arange(7 * 5).reshape(7, 5)
    b = ak.layout.NumpyArray(a)
    b = v1_to_v2(b)

    c1 = np.array([[4, 1], [1, 3], [0, 4]])
    c2 = np.array([[2, 2], [0, 1], [1, 3]])
    assert ak.to_list(b[c1, c2]) == ak.to_list(a[c1, c2])

    c = a % 2 == 0  # two dimensional
    assert ak.to_list(b[c]) == ak.to_list(a[c])
    assert ak.to_list(b[c,]) == ak.to_list(  # noqa: E231
        a[
            c,
        ]
    )
