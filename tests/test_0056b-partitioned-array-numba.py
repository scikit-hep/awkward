# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

numba = pytest.importorskip("numba")

awkward1_numba_arrayview = pytest.importorskip("awkward1._connect._numba.arrayview")

def test_view():
    aslist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    asarray = awkward1.repartition(awkward1.Array(aslist), 3)
    asview = awkward1_numba_arrayview.ArrayView.fromarray(asarray)

    for start in range(10):
        for stop in range(start, 10):
            asview.start = start
            asview.stop = stop
            assert awkward1.to_list(asview.toarray()) == aslist[start:stop]

    asarray = awkward1.repartition(awkward1.Array(aslist), [3, 2, 0, 1, 4])
    asview = awkward1_numba_arrayview.ArrayView.fromarray(asarray)

    for start in range(10):
        for stop in range(start, 10):
            asview.start = start
            asview.stop = stop
            assert awkward1.to_list(asview.toarray()) == aslist[start:stop]

    aslist = [[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]]
    asarray = awkward1.repartition(awkward1.Array(aslist), 3)
    asview = awkward1_numba_arrayview.ArrayView.fromarray(asarray)

    for start in range(5):
        for stop in range(start, 5):
            asview.start = start
            asview.stop = stop
            assert awkward1.to_list(asview.toarray()) == aslist[start:stop]

def test_boxing1():
    asnumpy = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert sys.getrefcount(asnumpy) == 2

    aslayout = awkward1.layout.NumpyArray(asnumpy)
    aspart = awkward1.repartition(aslayout, 3, highlevel=False)
    asarray = awkward1.Array(aspart)

    assert (sys.getrefcount(asnumpy), sys.getrefcount(aslayout), sys.getrefcount(aspart)) == (3, 2, 3)

    @numba.njit
    def f1(x):
        return 3.14

    for i in range(10):
        f1(asarray)
        assert (sys.getrefcount(asnumpy), sys.getrefcount(aslayout), sys.getrefcount(aspart)) == (3, 2, 3)

    del asarray
    del aspart
    del aslayout
    import gc
    gc.collect()
    assert sys.getrefcount(asnumpy) == 2

def test_boxing2():
    asnumpy = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert sys.getrefcount(asnumpy) == 2

    aslayout = awkward1.layout.NumpyArray(asnumpy)
    aspart = awkward1.repartition(aslayout, 3, highlevel=False)
    asarray = awkward1.Array(aspart)

    assert (sys.getrefcount(asnumpy), sys.getrefcount(aslayout), sys.getrefcount(aspart)) == (3, 2, 3)

    @numba.njit
    def f2(x):
        return x

    for i in range(10):
        out = f2(asarray)

        assert isinstance(out.layout, awkward1.partition.PartitionedArray)
        assert awkward1.to_list(out) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert (sys.getrefcount(asnumpy), sys.getrefcount(aslayout), sys.getrefcount(aspart)) == (3, 2, 3)

    del out
    del asarray
    del aspart
    del aslayout
    import gc
    gc.collect()
    assert sys.getrefcount(asnumpy) == 2

def test_boxing3():
    asnumpy = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert sys.getrefcount(asnumpy) == 2

    aslayout = awkward1.layout.NumpyArray(asnumpy)
    aspart = awkward1.repartition(aslayout, 3, highlevel=False)
    asarray = awkward1.Array(aspart)

    assert (sys.getrefcount(asnumpy), sys.getrefcount(aslayout), sys.getrefcount(aspart)) == (3, 2, 3)

    @numba.njit
    def f3(x):
        return x, x

    for i in range(10):
        out1, out2 = f3(asarray)
        assert isinstance(out1.layout, awkward1.partition.PartitionedArray)
        assert isinstance(out2.layout, awkward1.partition.PartitionedArray)
        assert awkward1.to_list(out1) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert awkward1.to_list(out2) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert (sys.getrefcount(asnumpy), sys.getrefcount(aslayout), sys.getrefcount(aspart)) == (3, 2, 3)

    del out1
    del out2
    del asarray
    del aspart
    del aslayout
    import gc
    gc.collect()
    assert sys.getrefcount(asnumpy) == 2

def test_getitem():
    array = awkward1.repartition(awkward1.Array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]), 3)

    @numba.njit
    def f1(x, i):
        return x[i]

    assert [f1(array, i) for i in range(10)] == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

