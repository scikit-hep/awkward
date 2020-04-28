# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import pytest
import numpy

import awkward1

numba = pytest.importorskip("numba")
awkward1_connect_numba_arrayview = pytest.importorskip("awkward1._connect._numba.arrayview")

def test_numpyarray():
    layout = awkward1.from_iter([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], highlevel=False)

    numbatype = awkward1._connect._numba.arrayview.tonumbatype(layout.form)
    assert numba.typeof(layout).name == numbatype.name

    lookup1 = awkward1_connect_numba_arrayview.Lookup(layout)
    lookup2 = awkward1_connect_numba_arrayview.Lookup(layout.form)
    numbatype.form_fill(0, layout, lookup2)

    assert numpy.array_equal(lookup1.arrayptrs, lookup2.arrayptrs)
    assert numpy.array_equal(lookup1.sharedptrs == -1, lookup2.sharedptrs == -1)

    counter = [0]
    def materialize():
        counter[0] += 1
        return layout

    generator = awkward1.virtual.ArrayGenerator(materialize, form=layout.form, length=len(layout))
    virtualarray = awkward1.layout.VirtualArray(generator)

    lookup3 = awkward1_connect_numba_arrayview.Lookup(virtualarray)
    assert len(lookup1.arrayptrs) + 3 == len(lookup3.arrayptrs)

    array = awkward1.Array(virtualarray)
    array.numba_type
    assert counter[0] == 0

    @numba.njit
    def f1(x):
        return x[5]

    assert f1(array) == 5.5
    assert counter[0] == 1

    assert f1(array) == 5.5
    assert counter[0] == 1

def test_listarray():
    layout = awkward1.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], highlevel=False)

    numbatype = awkward1._connect._numba.arrayview.tonumbatype(layout.form)
    assert numba.typeof(layout).name == numbatype.name

    lookup1 = awkward1_connect_numba_arrayview.Lookup(layout)
    lookup2 = awkward1_connect_numba_arrayview.Lookup(layout.form)
    numbatype.form_fill(0, layout, lookup2)

    assert numpy.array_equal(lookup1.arrayptrs, lookup2.arrayptrs)
    assert numpy.array_equal(lookup1.sharedptrs == -1, lookup2.sharedptrs == -1)

    counter = [0]
    def materialize():
        counter[0] += 1
        return layout

    generator = awkward1.virtual.ArrayGenerator(materialize, form=layout.form, length=len(layout))
    virtualarray = awkward1.layout.VirtualArray(generator)

    lookup3 = awkward1_connect_numba_arrayview.Lookup(virtualarray)
    assert len(lookup1.arrayptrs) + 3 == len(lookup3.arrayptrs)

    array = awkward1.Array(virtualarray)
    array.numba_type
    assert counter[0] == 0

    @numba.njit
    def f3(x):
        return x

    print(f3(array).layout)
    assert isinstance(f3(array).layout, awkward1.layout.VirtualArray)
    assert counter[0] == 0

    @numba.njit
    def f1(x):
        return x[2][1]

    assert f1(array) == 5.5
    assert counter[0] == 1

    assert f1(array) == 5.5
    assert counter[0] == 1

    @numba.njit
    def f2(x):
        return x[2]

    assert awkward1.to_list(f2(array)) == [4.4, 5.5]
    assert counter[0] == 1

    assert awkward1.to_list(f2(array)) == [4.4, 5.5]
    assert counter[0] == 1

    assert awkward1.to_list(f3(array)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
