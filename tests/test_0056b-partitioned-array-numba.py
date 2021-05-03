# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import sys

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

numba = pytest.importorskip("numba")

ak_numba = pytest.importorskip("awkward._connect._numba")
ak_numba_arrayview = pytest.importorskip("awkward._connect._numba.arrayview")

ak_numba.register_and_check()


def test_view():
    aslist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    asarray = ak.repartition(ak.Array(aslist), 3)
    asview = ak_numba_arrayview.ArrayView.fromarray(asarray)

    for start in range(10):
        for stop in range(start, 10):
            asview.start = start
            asview.stop = stop
            assert ak.to_list(asview.toarray()) == aslist[start:stop]

    asarray = ak.repartition(ak.Array(aslist), [3, 2, 0, 1, 4])
    asview = ak_numba_arrayview.ArrayView.fromarray(asarray)

    for start in range(10):
        for stop in range(start, 10):
            asview.start = start
            asview.stop = stop
            assert ak.to_list(asview.toarray()) == aslist[start:stop]

    aslist = [[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]]
    asarray = ak.repartition(ak.Array(aslist), 3)
    asview = ak_numba_arrayview.ArrayView.fromarray(asarray)

    for start in range(5):
        for stop in range(start, 5):
            asview.start = start
            asview.stop = stop
            assert ak.to_list(asview.toarray()) == aslist[start:stop]


def test_boxing1():
    asnumpy = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert sys.getrefcount(asnumpy) == 2

    aslayout = ak.layout.NumpyArray(asnumpy)
    aspart = ak.repartition(aslayout, 3, highlevel=False)
    asarray = ak.Array(aspart)
    aspart = asarray._layout

    assert (
        sys.getrefcount(asnumpy),
        sys.getrefcount(aslayout),
        sys.getrefcount(aspart),
    ) == (3, 2, 3)

    @numba.njit
    def f1(x):
        return 3.14

    for _ in range(5):
        f1(asarray)
        assert (
            sys.getrefcount(asnumpy),
            sys.getrefcount(aslayout),
            sys.getrefcount(aspart),
        ) == (3, 2, 3)

    del asarray
    del aspart
    del aslayout
    import gc

    gc.collect()
    assert sys.getrefcount(asnumpy) == 2


def test_boxing2():
    asnumpy = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert sys.getrefcount(asnumpy) == 2

    aslayout = ak.layout.NumpyArray(asnumpy)
    aspart = ak.repartition(aslayout, 3, highlevel=False)
    asarray = ak.Array(aspart)
    aspart = asarray._layout

    assert (
        sys.getrefcount(asnumpy),
        sys.getrefcount(aslayout),
        sys.getrefcount(aspart),
    ) == (3, 2, 3)

    @numba.njit
    def f2(x):
        return x

    for _ in range(10):
        out = f2(asarray)

        assert isinstance(out.layout, ak.partition.PartitionedArray)
        assert ak.to_list(out) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert (
            sys.getrefcount(asnumpy),
            sys.getrefcount(aslayout),
            sys.getrefcount(aspart),
        ) == (3, 2, 3)

    del out
    del asarray
    del aspart
    del aslayout
    import gc

    gc.collect()
    assert sys.getrefcount(asnumpy) == 2


def test_boxing3():
    asnumpy = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert sys.getrefcount(asnumpy) == 2

    aslayout = ak.layout.NumpyArray(asnumpy)
    aspart = ak.repartition(aslayout, 3, highlevel=False)
    asarray = ak.Array(aspart)
    aspart = asarray._layout

    assert (
        sys.getrefcount(asnumpy),
        sys.getrefcount(aslayout),
        sys.getrefcount(aspart),
    ) == (3, 2, 3)

    @numba.njit
    def f3(x):
        return x, x

    for _ in range(10):
        out1, out2 = f3(asarray)
        assert isinstance(out1.layout, ak.partition.PartitionedArray)
        assert isinstance(out2.layout, ak.partition.PartitionedArray)
        assert ak.to_list(out1) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert ak.to_list(out2) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert (
            sys.getrefcount(asnumpy),
            sys.getrefcount(aslayout),
            sys.getrefcount(aspart),
        ) == (3, 2, 3)

    del out1
    del out2
    del asarray
    del aspart
    del aslayout
    import gc

    gc.collect()
    assert sys.getrefcount(asnumpy) == 2


def test_getitem_1a():
    array = ak.repartition(
        ak.Array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]), 3
    )

    @numba.njit
    def f1(x, i):
        return x[i]

    assert [f1(array, i) for i in range(10)] == [
        0.0,
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        6.6,
        7.7,
        8.8,
        9.9,
    ]
    assert [f1(array, -i) for i in range(1, 11)] == [
        9.9,
        8.8,
        7.7,
        6.6,
        5.5,
        4.4,
        3.3,
        2.2,
        1.1,
        0.0,
    ]


def test_getitem_1b():
    asnumpy = np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    array = ak.repartition(ak.Array(asnumpy), 3)

    assert sys.getrefcount(asnumpy) == 3

    @numba.njit
    def f2(x, i1, i2):
        out = x[i1:i2]
        return out

    assert isinstance(f2(array, 0, 10).layout, ak.partition.PartitionedArray)
    assert isinstance(f2(array, 4, 5).layout, ak.partition.PartitionedArray)
    assert isinstance(f2(array, 5, 5).layout, ak.partition.PartitionedArray)

    for start in range(-10, 10):
        for stop in range(-10, 10):
            assert (
                ak.to_list(f2(array, start, stop))
                == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9][start:stop]
            )

    assert sys.getrefcount(asnumpy) == 3

    del array
    assert sys.getrefcount(asnumpy) == 2


def test_getitem_2():
    aslist = [
        {"x": 0.0, "y": []},
        {"x": 1.1, "y": [1]},
        {"x": 2.2, "y": [2, 2]},
        {"x": 3.3, "y": [3, 3, 3]},
        {"x": 4.4, "y": [4, 4, 4, 4]},
        {"x": 5.5, "y": [5, 5, 5]},
        {"x": 6.6, "y": [6, 6]},
        {"x": 7.7, "y": [7]},
        {"x": 8.8, "y": []},
    ]
    asarray = ak.repartition(ak.Array(aslist), 2)

    @numba.njit
    def f3a(x):
        return x["x"]

    assert ak.to_list(f3a(asarray)) == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]

    @numba.njit
    def f3b(x):
        return x.x

    assert ak.to_list(f3b(asarray)) == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]

    @numba.njit
    def f4a(x):
        return x["y"]

    assert ak.to_list(f4a(asarray)) == [
        [],
        [1],
        [2, 2],
        [3, 3, 3],
        [4, 4, 4, 4],
        [5, 5, 5],
        [6, 6],
        [7],
        [],
    ]

    @numba.njit
    def f4b(x):
        return x.y

    assert ak.to_list(f4b(asarray)) == [
        [],
        [1],
        [2, 2],
        [3, 3, 3],
        [4, 4, 4, 4],
        [5, 5, 5],
        [6, 6],
        [7],
        [],
    ]

    @numba.njit
    def f5a(x, i):
        return x["x"][i]

    assert [f5a(asarray, i) for i in range(-9, 9)]

    @numba.njit
    def f5b(x, i):
        return x[i]["x"]

    assert [f5b(asarray, i) for i in range(-9, 9)]

    @numba.njit
    def f5c(x, i):
        return x.x[i]

    assert [f5c(asarray, i) for i in range(-9, 9)]

    @numba.njit
    def f5d(x, i):
        return x[i].x

    assert [f5d(asarray, i) for i in range(-9, 9)]

    @numba.njit
    def f6a(x, i):
        return x["y"][i]

    assert ak.to_list(f6a(asarray, 6)) == [6, 6]
    assert ak.to_list(f6a(asarray, -3)) == [6, 6]

    @numba.njit
    def f6b(x, i):
        return x[i]["y"]

    assert ak.to_list(f6b(asarray, 6)) == [6, 6]
    assert ak.to_list(f6b(asarray, -3)) == [6, 6]

    @numba.njit
    def f6c(x, i):
        return x.y[i]

    assert ak.to_list(f6c(asarray, 6)) == [6, 6]
    assert ak.to_list(f6c(asarray, -3)) == [6, 6]

    @numba.njit
    def f6d(x, i):
        return x[i].y

    assert ak.to_list(f6d(asarray, 6)) == [6, 6]
    assert ak.to_list(f6d(asarray, -3)) == [6, 6]


def test_len():
    array = ak.repartition(ak.Array([1.1, 2.2, 3.3, 4.4, 5.5]), 3)

    @numba.njit
    def f1(x):
        return len(x)

    assert f1(array) == 5

    aslist = [
        {"x": 0.0, "y": []},
        {"x": 1.1, "y": [1]},
        {"x": 2.2, "y": [2, 2]},
        {"x": 3.3, "y": [3, 3, 3]},
        {"x": 4.4, "y": [4, 4, 4, 4]},
        {"x": 5.5, "y": [5, 5, 5]},
        {"x": 6.6, "y": [6, 6]},
        {"x": 7.7, "y": [7]},
        {"x": 8.8, "y": []},
    ]
    asarray = ak.repartition(ak.Array(aslist), 2)

    assert f1(asarray) == 9


def test_iter():
    asnumpy = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    assert sys.getrefcount(asnumpy) == 2

    array = ak.repartition(ak.Array(asnumpy), 3)

    assert sys.getrefcount(asnumpy) == 3

    @numba.njit
    def f1(x):
        out = 0
        for xi in x:
            out += xi
        return out

    for _ in range(10):
        assert f1(array) == 45
        assert sys.getrefcount(asnumpy) == 3

    del array
    assert sys.getrefcount(asnumpy) == 2

    aslist = [
        {"x": 0.0, "y": []},
        {"x": 1.1, "y": [1]},
        {"x": 2.2, "y": [2, 2]},
        {"x": 3.3, "y": [3, 3, 3]},
        {"x": 4.4, "y": [4, 4, 4, 4]},
        {"x": 5.5, "y": [5, 5, 5]},
        {"x": 6.6, "y": [6, 6]},
        {"x": 7.7, "y": [7]},
        {"x": 8.8, "y": []},
    ]
    asarray = ak.repartition(ak.Array(aslist), 2)

    @numba.njit
    def f2(x):
        i = 0
        for xi in x:
            if i == 6:
                return xi["y"]
            i += 1

    assert ak.to_list(f2(asarray)) == [6, 6]

    @numba.njit
    def f3(x):
        i = 0
        for xi in x:
            if i == 6:
                return xi
            i += 1

    assert ak.to_list(f3(asarray)) == {"x": 6.6, "y": [6, 6]}
