# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test_numpyarray_getitem_bystrides():
    a = np.arange(10)
    b = ak.contents.NumpyArray(a)

    assert b[3] == a[3]
    assert b[-3] == a[-3]
    assert to_list(b[()]) == to_list(a[()])
    assert to_list(b[slice(None)]) == to_list(a[slice(None)])
    assert to_list(b[slice(3, 7)]) == to_list(a[slice(3, 7)])
    assert to_list(b[slice(3, 100)]) == to_list(a[slice(3, 100)])
    assert to_list(b[slice(-100, 7)]) == to_list(a[slice(-100, 7)])
    assert to_list(b[slice(3, -3)]) == to_list(a[slice(3, -3)])
    assert to_list(b[slice(1, 7, 2)]) == to_list(a[slice(1, 7, 2)])
    assert to_list(b[slice(-8, 7, 2)]) == to_list(a[slice(-8, 7, 2)])
    assert to_list(b[slice(None, 7, 2)]) == to_list(a[slice(None, 7, 2)])
    assert to_list(b[slice(None, -3, 2)]) == to_list(a[slice(None, -3, 2)])
    assert to_list(b[slice(8, None, -1)]) == to_list(a[slice(8, None, -1)])
    assert to_list(b[slice(8, None, -2)]) == to_list(a[slice(8, None, -2)])
    assert to_list(b[slice(-2, None, -2)]) == to_list(a[slice(-2, None, -2)])

    a = np.arange(7 * 5).reshape(7, 5)
    b = ak.contents.NumpyArray(a)

    assert to_list(b[()]) == to_list(a[()])
    assert to_list(b[3]) == to_list(a[3])
    assert to_list(b[(3, 2)]) == to_list(a[3, 2])
    assert to_list(b[slice(1, 4)]) == to_list(a[slice(1, 4)])
    assert to_list(b[(3, slice(1, 4))]) == to_list(a[3, slice(1, 4)])
    assert to_list(b[(slice(1, 4), 3)]) == to_list(a[slice(1, 4), 3])
    assert to_list(b[(slice(1, 4), slice(2, None))]) == to_list(
        a[slice(1, 4), slice(2, None)]
    )
    assert to_list(b[(slice(None, None, -1), slice(2, None))]) == to_list(
        a[slice(None, None, -1), slice(2, None)]
    )

    assert to_list(b[:, np.newaxis, :]) == to_list(a[:, np.newaxis, :])
    assert to_list(b[np.newaxis, :, np.newaxis, :, np.newaxis]) == to_list(
        a[np.newaxis, :, np.newaxis, :, np.newaxis]
    )

    assert to_list(b[Ellipsis, 3]) == to_list(a[Ellipsis, 3])
    assert to_list(b[Ellipsis, 3, 2]) == to_list(a[Ellipsis, 3, 2])
    assert to_list(b[3, Ellipsis]) == to_list(a[3, Ellipsis])
    assert to_list(b[3, 2, Ellipsis]) == to_list(a[3, 2, Ellipsis])


def test_numpyarray_getitem_next():
    a = np.arange(10)
    b = ak.contents.NumpyArray(a)
    c = np.array([7, 3, 3, 5])

    assert to_list(b[c]) == to_list(a[c])
    assert b.to_typetracer()[c].form == b[c].form
    c = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    assert to_list(b[c]) == to_list(a[c])
    assert b.to_typetracer()[c].form == b[c].form
    c = np.array([False, False, False, True, True, True, False, True, False, True])
    assert to_list(b[c]) == to_list(a[c])
    assert b.to_typetracer()[c].form == b[c].form
    c = np.array([], dtype=np.int64)
    assert to_list(b[c]) == to_list(a[c])
    assert b.to_typetracer()[c].form == b[c].form

    a = np.arange(10 * 3).reshape(10, 3)
    b = ak.contents.NumpyArray(a)
    c = np.array([7, 3, 3, 5])

    assert to_list(b[c]) == to_list(a[c])
    assert b.to_typetracer()[c].form == b[c].form
    c = np.array([False, False, False, True, True, True, False, True, False, True])
    assert to_list(b[c]) == to_list(a[c])
    assert b.to_typetracer()[c].form == b[c].form
    c = np.array([], dtype=np.int64)
    assert to_list(b[c]) == to_list(a[c])
    assert b.to_typetracer()[c].form == b[c].form

    a = np.arange(7 * 5).reshape(7, 5)
    b = ak.contents.NumpyArray(a)

    c1 = np.array([], np.int64)
    c2 = np.array([], np.int64)

    assert to_list(b[c1, c2]) == to_list(a[c1, c2])
    assert a[c1, c2].ndim == 1
    assert b.to_typetracer()[c1, c2].form == b[c1, c2].form

    a = np.arange(7 * 5).reshape(7, 5)
    b = ak.contents.NumpyArray(a)
    c1 = np.array([4, 1, 1, 3])
    c2 = np.array([2, 2, 0, 1])
    assert to_list(b[c1, c2]) == to_list(a[c1, c2])
    assert b.to_typetracer()[c1, c2].form == b[c1, c2].form

    c = np.array([False, False, True, True, False, True, True])
    assert to_list(b[c]) == to_list(a[c])
    assert b.to_typetracer()[c].form == b[c].form

    c = np.array([], dtype=np.int64)
    assert to_list(b[c]) == to_list(a[c])
    c1 = np.array([], dtype=np.int64)
    c2 = np.array([], dtype=np.int64)
    assert to_list(b[c1, c2]) == to_list(a[c1, c2])
    assert b.to_typetracer()[c1, c2].form == b[c1, c2].form

    a = np.arange(7 * 5).reshape(7, 5)
    b = ak.contents.NumpyArray(a)
    c = np.array([2, 0, 0, 1])
    assert to_list(b[1:4, c]) == to_list(a[1:4, c])
    assert to_list(b[c, 1:4]) == to_list(a[c, 1:4])
    assert to_list(b[1:4, np.newaxis, c]) == to_list(a[1:4, np.newaxis, c])
    assert to_list(b[c, np.newaxis, 1:4]) == to_list(a[c, np.newaxis, 1:4])
    assert to_list(b[1:4, np.newaxis, np.newaxis, c]) == to_list(
        a[1:4, np.newaxis, np.newaxis, c]
    )
    assert to_list(b[c, np.newaxis, np.newaxis, 1:4]) == to_list(
        a[c, np.newaxis, np.newaxis, 1:4]
    )
    assert to_list(b[Ellipsis, c]) == to_list(a[Ellipsis, c])
    assert to_list(b[c, Ellipsis]) == to_list(a[c, Ellipsis])


def test_numpyarray_getitem_next_2():
    a = np.arange(7 * 5).reshape(7, 5)
    b = ak.contents.NumpyArray(a)

    c1 = np.array([[4, 1], [1, 3], [0, 4]])
    c2 = np.array([[2, 2], [0, 1], [1, 3]])
    assert to_list(b[c1, c2]) == to_list(a[c1, c2])

    c = a % 2 == 0  # two dimensional
    assert to_list(b[c]) == to_list(a[c])
    assert to_list(b[c,]) == to_list(  # noqa: E231
        a[
            c,
        ]
    )
