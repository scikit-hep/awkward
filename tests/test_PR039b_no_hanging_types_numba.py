# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

py27 = (sys.version_info[0] < 3)

pytest.skip("Disabling Numba until the rewrite is done.", allow_module_level=True)

numba = pytest.importorskip("numba")

class D(awkward1.highlevel.Array):
    pass

def test_numpyarray():
    array1 = awkward1.layout.NumpyArray(numpy.arange(2*3*5, dtype=numpy.int64).reshape(2, 3, 5), parameters={"__record__": "D", "__typestr__": "D[int64]"})

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.type) == "3 * 5 * D[int64]"
    assert repr(array2[0].type) == "5 * D[int64]"
    assert repr(array2[0, 0].type) == "D[int64]"
    assert array2[-1, -1, -1] == 29

def test_regulararray():
    array1 = awkward1.layout.RegularArray(awkward1.layout.NumpyArray(numpy.arange(10, dtype=numpy.int64)), 5, parameters={"__record__": "D", "__typestr__": "D[5 * int64]"})

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.type) == "D[5 * int64]"

def test_listoffsetarray():
    array1 = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64)), awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64)), parameters={"__record__": "D", "__typestr__": "D[var * int64]"})

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.type) == "D[var * int64]"

def test_listarray():
    array1 = awkward1.layout.ListArray64(awkward1.layout.Index64(numpy.array([0, 3, 3], dtype=numpy.int64)), awkward1.layout.Index64(numpy.array([3, 3, 5], dtype=numpy.int64)), awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64)), parameters={"__record__": "D", "__typestr__": "D[var * int64]"})

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.type) == "D[var * int64]"

def test_recordarray():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3], dtype=numpy.int64))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3], dtype=numpy.float64))
    array1 = awkward1.layout.RecordArray({"one": content1, "two": content2}, parameters={"__record__": "D", "__typestr__": "D[{\"one\": int64, \"two\": float64}]"})

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.type) in ('D[{"one": int64, "two": float64}]', ' D[{"two": float64, "one": int64}]')
    assert repr(array2[0].type) in ('D[{"one": int64, "two": float64}]', 'D[{"two": float64, "one": int64}]')
