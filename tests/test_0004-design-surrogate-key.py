# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import gc

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward1 as ak  # noqa: F401


def test_refcount1():
    i = np.arange(12, dtype="i4").reshape(3, 4)
    assert sys.getrefcount(i) == 2

    i2 = ak.layout.Identities32(
        ak.layout.Identities32.newref(), [(0, "hey"), (1, "there")], i
    )
    assert (sys.getrefcount(i), sys.getrefcount(i2)) == (3, 2)

    tmp = np.asarray(i2)
    assert tmp.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

    assert (sys.getrefcount(i), sys.getrefcount(i2)) == (
        3,
        2 + (2 if ak._util.py27 else 1),
    )

    del tmp
    assert (sys.getrefcount(i), sys.getrefcount(i2)) == (3, 2)

    tmp2 = i2.array
    assert tmp2.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

    assert (sys.getrefcount(i), sys.getrefcount(i2)) == (
        3,
        2 + (2 if ak._util.py27 else 1),
    )

    del tmp2
    assert (sys.getrefcount(i), sys.getrefcount(i2)) == (3, 2)

    del i2
    assert sys.getrefcount(i) == 2


def test_refcount2():
    i = np.arange(6, dtype="i4").reshape(3, 2)
    i2 = ak.layout.Identities32(ak.layout.Identities32.newref(), [], i)
    x = np.arange(12).reshape(3, 4)
    x2 = ak.layout.NumpyArray(x)
    x2.identities = i2
    del i
    del i2
    del x
    i3 = x2.identities
    del x2
    gc.collect()
    assert np.asarray(i3).tolist() == [[0, 1], [2, 3], [4, 5]]
    del i3
    gc.collect()


def test_refcount3():
    i = np.arange(6, dtype="i4").reshape(3, 2)
    i2 = ak.layout.Identities32(ak.layout.Identities32.newref(), [], i)
    x = np.arange(12).reshape(3, 4)
    x2 = ak.layout.NumpyArray(x)
    x2.identities = i2
    del i2
    assert sys.getrefcount(i) == 3
    x2.identities = None
    assert sys.getrefcount(i) == 2


def test_numpyarray_setidentities():
    x = np.arange(160).reshape(40, 4)
    x2 = ak.layout.NumpyArray(x)
    x2.setidentities()
    assert np.asarray(x2.identities).tolist() == np.arange(40).reshape(40, 1).tolist()


def test_listoffsetarray_setidentities():
    content = ak.layout.NumpyArray(np.arange(10))
    offsets = ak.layout.Index32(np.array([0, 3, 3, 5, 10], dtype="i4"))
    jagged = ak.layout.ListOffsetArray32(offsets, content)
    jagged.setidentities()
    assert np.asarray(jagged.identities).tolist() == [[0], [1], [2], [3]]
    assert np.asarray(jagged.content.identities).tolist() == [
        [0, 0],
        [0, 1],
        [0, 2],
        [2, 0],
        [2, 1],
        [3, 0],
        [3, 1],
        [3, 2],
        [3, 3],
        [3, 4],
    ]

    assert np.asarray(jagged.content[3:7].identities).tolist() == [
        [2, 0],
        [2, 1],
        [3, 0],
        [3, 1],
    ]
    assert np.asarray(jagged[0].identities).tolist() == [[0, 0], [0, 1], [0, 2]]
    assert np.asarray(jagged[1].identities).tolist() == []
    assert np.asarray(jagged[2].identities).tolist() == [[2, 0], [2, 1]]
    assert np.asarray(jagged[3].identities).tolist() == [
        [3, 0],
        [3, 1],
        [3, 2],
        [3, 3],
        [3, 4],
    ]
    assert np.asarray(jagged[1:3].identities).tolist() == [[1], [2]]


def test_setidentities_none():
    offsets = ak.layout.Index32(np.array([0, 2, 2, 3], "i4"))
    content = ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3]))
    array = ak.layout.ListOffsetArray32(offsets, content)
    assert array.identities is None
    assert array.content.identities is None
    array.identities = None
    assert array.identities is None
    assert array.content.identities is None
    repr(array)
    array.setidentities()
    repr(array)
    assert array.identities is not None
    assert array.content.identities is not None
    array.identities = None
    assert array.identities is None
    assert array.content.identities is None


def test_setidentities_constructor():
    offsets = ak.layout.Index32(np.array([0, 2, 2, 3], "i4"))
    content = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3]),
        identities=ak.layout.Identities32(
            ak.layout.Identities32.newref(),
            [],
            np.array([[0, 0], [0, 1], [2, 0]], dtype="i4"),
        ),
    )
    array = ak.layout.ListOffsetArray32(
        offsets,
        content,
        identities=ak.layout.Identities32(
            ak.layout.Identities32.newref(), [], np.array([[0], [1], [2]], dtype="i4")
        ),
    )
    assert np.asarray(array.identities).tolist() == [[0], [1], [2]]
    assert np.asarray(array.content.identities).tolist() == [[0, 0], [0, 1], [2, 0]]
