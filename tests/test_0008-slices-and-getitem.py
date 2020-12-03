# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_slice():
    assert ak._ext._slice_tostring(3) == "[3]"
    assert ak._ext._slice_tostring(slice(None)) == "[:]"
    assert ak._ext._slice_tostring(slice(10)) == "[:10]"
    assert ak._ext._slice_tostring(slice(1, 2)) == "[1:2]"
    assert ak._ext._slice_tostring(slice(1, None)) == "[1:]"
    assert ak._ext._slice_tostring(slice(None, None, 2)) == "[::2]"
    assert ak._ext._slice_tostring(slice(1, 2, 3)) == "[1:2:3]"
    if not ak._util.py27:
        assert ak._ext._slice_tostring(Ellipsis) == "[...]"
    assert ak._ext._slice_tostring(np.newaxis) == "[newaxis]"
    assert ak._ext._slice_tostring(None) == "[newaxis]"
    assert ak._ext._slice_tostring([1, 2, 3]) == "[array([1, 2, 3])]"
    assert (
        ak._ext._slice_tostring(np.array([[1, 2], [3, 4], [5, 6]]))
        == "[array([[1, 2], [3, 4], [5, 6]])]"
    )
    assert (
        ak._ext._slice_tostring(np.array([1, 2, 3, 4, 5, 6])[::-2])
        == "[array([6, 4, 2])]"
    )
    a = np.arange(3 * 5).reshape(3, 5)[1::, ::-2]
    assert ak._ext._slice_tostring(a) == "[array(" + str(a.tolist()) + ")]"
    a = np.arange(3 * 5).reshape(3, 5)[::-1, ::2]
    assert ak._ext._slice_tostring(a) == "[array(" + str(a.tolist()) + ")]"
    assert (
        ak._ext._slice_tostring([True, True, False, False, True])
        == "[array([0, 1, 4])]"
    )
    assert (
        ak._ext._slice_tostring([[True, True], [False, False], [True, False]])
        == "[array([0, 0, 2]), array([0, 1, 0])]"
    )
    assert ak._ext._slice_tostring(()) == "[]"
    assert ak._ext._slice_tostring((3,)) == "[3]"
    assert ak._ext._slice_tostring((3, slice(1, 2, 3))) == "[3, 1:2:3]"
    assert ak._ext._slice_tostring((slice(None), [1, 2, 3])) == "[:, array([1, 2, 3])]"
    assert ak._ext._slice_tostring(([1, 2, 3], slice(None))) == "[array([1, 2, 3]), :]"
    assert (
        ak._ext._slice_tostring((slice(None), [True, True, False, False, True]))
        == "[:, array([0, 1, 4])]"
    )
    assert (
        ak._ext._slice_tostring(
            (slice(None), [[True, True], [False, False], [True, False]])
        )
        == "[:, array([0, 0, 2]), array([0, 1, 0])]"
    )
    assert (
        ak._ext._slice_tostring(
            ([[True, True], [False, False], [True, False]], slice(None))
        )
        == "[array([0, 0, 2]), array([0, 1, 0]), :]"
    )

    with pytest.raises(ValueError):
        ak._ext._slice_tostring(np.array([1.1, 2.2, 3.3]))
    assert (
        ak._ext._slice_tostring(np.array(["one", "two", "three"]))
        == '[["one", "two", "three"]]'
    )
    with pytest.raises(ValueError):
        ak._ext._slice_tostring(np.array([1, 2, 3, None, 4, 5]))

    assert (
        ak._ext._slice_tostring((123, [[1, 1], [2, 2], [3, 3]]))
        == "[array([[123, 123], [123, 123], [123, 123]]), array([[1, 1], [2, 2], [3, 3]])]"
    )
    assert (
        ak._ext._slice_tostring(([[1, 1], [2, 2], [3, 3]], 123))
        == "[array([[1, 1], [2, 2], [3, 3]]), array([[123, 123], [123, 123], [123, 123]])]"
    )
    assert (
        ak._ext._slice_tostring(([[100, 200, 300, 400]], [[1], [2], [3]]))
        == "[array([[100, 200, 300, 400], [100, 200, 300, 400], [100, 200, 300, 400]]), array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])]"
    )
    assert (
        ak._ext._slice_tostring(([[1], [2], [3]], [[100, 200, 300, 400]]))
        == "[array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]), array([[100, 200, 300, 400], [100, 200, 300, 400], [100, 200, 300, 400]])]"
    )

    with pytest.raises(ValueError):
        ak._ext._slice_tostring((3, slice(None), [[1], [2], [3]]))
    with pytest.raises(ValueError):
        ak._ext._slice_tostring(([[1, 2, 3, 4]], slice(None), [[1], [2], [3]]))
    with pytest.raises(ValueError):
        ak._ext._slice_tostring(
            (slice(None), 3, slice(None), [[1], [2], [3]], slice(None))
        )
    with pytest.raises(ValueError):
        ak._ext._slice_tostring(
            (slice(None), [[1, 2, 3, 4]], slice(None), [[1], [2], [3]], slice(None))
        )
    assert (
        ak._ext._slice_tostring((slice(None), 3, [[1], [2], [3]], slice(None)))
        == "[:, array([[3], [3], [3]]), array([[1], [2], [3]]), :]"
    )
    assert (
        ak._ext._slice_tostring(
            (slice(None), [[1, 2, 3, 4]], [[1], [2], [3]], slice(None))
        )
        == "[:, array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]), array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]), :]"
    )


def test_numpyarray_getitem_bystrides():
    a = np.arange(10)
    b = ak.layout.NumpyArray(a)
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


def test_numpyarray_contiguous():
    a = np.arange(10)[8::-2]
    b = ak.layout.NumpyArray(a)

    assert ak.to_list(b) == ak.to_list(a)
    assert ak.to_list(b.contiguous()) == ak.to_list(a)
    b = b.contiguous()
    assert ak.to_list(b) == ak.to_list(a)

    a = np.arange(7 * 5).reshape(7, 5)[::-1, ::2]
    b = ak.layout.NumpyArray(a)

    assert ak.to_list(b) == ak.to_list(a)
    assert ak.to_list(b.contiguous())
    b = b.contiguous()
    assert ak.to_list(b) == ak.to_list(a)


def test_numpyarray_getitem_next():
    a = np.arange(10)
    b = ak.layout.NumpyArray(a)
    c = np.array([7, 3, 3, 5])
    assert ak.to_list(b[c]) == ak.to_list(a[c])
    c = np.array([False, False, False, True, True, True, False, True, False, True])
    assert ak.to_list(b[c]) == ak.to_list(a[c])
    c = np.array([], dtype=int)
    assert ak.to_list(b[c]) == ak.to_list(a[c])

    a = np.arange(10 * 3).reshape(10, 3)
    b = ak.layout.NumpyArray(a)
    c = np.array([7, 3, 3, 5])
    assert ak.to_list(b[c]) == ak.to_list(a[c])
    c = np.array([False, False, False, True, True, True, False, True, False, True])
    assert ak.to_list(b[c]) == ak.to_list(a[c])
    c = np.array([], dtype=int)
    assert ak.to_list(b[c]) == ak.to_list(a[c])

    a = np.arange(7 * 5).reshape(7, 5)
    b = ak.layout.NumpyArray(a)
    c1 = np.array([4, 1, 1, 3])
    c2 = np.array([2, 2, 0, 1])
    assert ak.to_list(b[c1, c2]) == ak.to_list(a[c1, c2])
    c1 = np.array([[4, 1], [1, 3], [0, 4]])
    c2 = np.array([[2, 2], [0, 1], [1, 3]])
    assert ak.to_list(b[c1, c2]) == ak.to_list(a[c1, c2])
    c = np.array([False, False, True, True, False, True, True])
    assert ak.to_list(b[c]) == ak.to_list(a[c])
    c = a % 2 == 0  # two dimensional
    assert ak.to_list(b[c]) == ak.to_list(a[c])
    c = np.array([], dtype=int)
    assert ak.to_list(b[c]) == ak.to_list(a[c])
    c1 = np.array([], dtype=int)
    c2 = np.array([], dtype=int)
    assert ak.to_list(b[c1, c2]) == ak.to_list(a[c1, c2])

    a = np.arange(7 * 5).reshape(7, 5)
    b = ak.layout.NumpyArray(a)
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
