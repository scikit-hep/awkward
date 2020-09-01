# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import pytest
import numpy

import awkward1

numba = pytest.importorskip("numba")
awkward1_connect_numba_arrayview = pytest.importorskip("awkward1._connect._numba.arrayview")
awkward1_connect_numba_layout = pytest.importorskip("awkward1._connect._numba.layout")

def test_numpyarray():
    layout = awkward1.from_iter([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], highlevel=False)

    numbatype = awkward1._connect._numba.arrayview.tonumbatype(layout.form)
    assert awkward1_connect_numba_layout.typeof(layout).name == numbatype.name

    lookup1 = awkward1_connect_numba_arrayview.Lookup(layout)
    lookup2 = awkward1_connect_numba_arrayview.Lookup(layout.form)
    numbatype.form_fill(0, layout, lookup2)

    assert numpy.array_equal(lookup1.arrayptrs, lookup2.arrayptrs)
    assert numpy.array_equal(lookup1.sharedptrs == -1, lookup2.sharedptrs == -1)

    counter = [0]
    def materialize():
        counter[0] += 1
        return layout

    generator = awkward1.layout.ArrayGenerator(materialize, form=layout.form, length=len(layout))
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
    for case in "ListOffsetArray", "ListArray":
        layout = awkward1.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], highlevel=False)
        if case == "ListArray":
            layout = layout[[0, 1, 2, 3, 4]]

        numbatype = awkward1._connect._numba.arrayview.tonumbatype(layout.form)
        assert awkward1_connect_numba_layout.typeof(layout).name == numbatype.name

        lookup1 = awkward1_connect_numba_arrayview.Lookup(layout)
        lookup2 = awkward1_connect_numba_arrayview.Lookup(layout.form)
        numbatype.form_fill(0, layout, lookup2)

        assert numpy.array_equal(lookup1.arrayptrs, lookup2.arrayptrs)
        assert numpy.array_equal(lookup1.sharedptrs == -1, lookup2.sharedptrs == -1)

        counter = [0]
        def materialize():
            counter[0] += 1
            return layout

        generator = awkward1.layout.ArrayGenerator(materialize, form=layout.form, length=len(layout))
        virtualarray = awkward1.layout.VirtualArray(generator)

        lookup3 = awkward1_connect_numba_arrayview.Lookup(virtualarray)
        assert len(lookup1.arrayptrs) + 3 == len(lookup3.arrayptrs)

        array = awkward1.Array(virtualarray)
        array.numba_type
        assert counter[0] == 0

        @numba.njit
        def f3(x):
            return x

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

def test_regulararray():
    layout = awkward1.from_numpy(numpy.array([[1, 2, 3], [4, 5, 6]]), regulararray=True, highlevel=False)

    numbatype = awkward1._connect._numba.arrayview.tonumbatype(layout.form)
    assert awkward1_connect_numba_layout.typeof(layout).name == numbatype.name

    lookup1 = awkward1_connect_numba_arrayview.Lookup(layout)
    lookup2 = awkward1_connect_numba_arrayview.Lookup(layout.form)
    numbatype.form_fill(0, layout, lookup2)

    assert numpy.array_equal(lookup1.arrayptrs, lookup2.arrayptrs)
    assert numpy.array_equal(lookup1.sharedptrs == -1, lookup2.sharedptrs == -1)

    counter = [0]
    def materialize():
        counter[0] += 1
        return layout

    generator = awkward1.layout.ArrayGenerator(materialize, form=layout.form, length=len(layout))
    virtualarray = awkward1.layout.VirtualArray(generator)

    lookup3 = awkward1_connect_numba_arrayview.Lookup(virtualarray)
    assert len(lookup1.arrayptrs) + 3 == len(lookup3.arrayptrs)

    array = awkward1.Array(virtualarray)
    array.numba_type
    assert counter[0] == 0

    @numba.njit
    def f3(x):
        return x

    assert isinstance(f3(array).layout, awkward1.layout.VirtualArray)
    assert counter[0] == 0

    @numba.njit
    def f1(x):
        return x[1][1]

    assert f1(array) == 5
    assert counter[0] == 1

    assert f1(array) == 5
    assert counter[0] == 1

    @numba.njit
    def f2(x):
        return x[1]

    assert awkward1.to_list(f2(array)) == [4, 5, 6]
    assert counter[0] == 1

    assert awkward1.to_list(f2(array)) == [4, 5, 6]
    assert counter[0] == 1

    assert awkward1.to_list(f3(array)) == [[1, 2, 3], [4, 5, 6]]

def test_indexedarray():
    layout = awkward1.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], highlevel=False)
    layout = awkward1.layout.IndexedArray64(awkward1.layout.Index64(numpy.array([4, 3, 2, 1, 0], dtype=numpy.int64)), layout)

    numbatype = awkward1._connect._numba.arrayview.tonumbatype(layout.form)
    assert awkward1_connect_numba_layout.typeof(layout).name == numbatype.name

    lookup1 = awkward1_connect_numba_arrayview.Lookup(layout)
    lookup2 = awkward1_connect_numba_arrayview.Lookup(layout.form)
    numbatype.form_fill(0, layout, lookup2)

    assert numpy.array_equal(lookup1.arrayptrs, lookup2.arrayptrs)
    assert numpy.array_equal(lookup1.sharedptrs == -1, lookup2.sharedptrs == -1)

    counter = [0]
    def materialize():
        counter[0] += 1
        return layout

    generator = awkward1.layout.ArrayGenerator(materialize, form=layout.form, length=len(layout))
    virtualarray = awkward1.layout.VirtualArray(generator)

    lookup3 = awkward1_connect_numba_arrayview.Lookup(virtualarray)
    assert len(lookup1.arrayptrs) + 3 == len(lookup3.arrayptrs)

    array = awkward1.Array(virtualarray)
    array.numba_type
    assert counter[0] == 0

    @numba.njit
    def f3(x):
        return x

    assert isinstance(f3(array).layout, awkward1.layout.VirtualArray)
    assert counter[0] == 0

    @numba.njit
    def f1(x):
        return x[4][1]

    assert f1(array) == 2.2
    assert counter[0] == 1

    assert f1(array) == 2.2
    assert counter[0] == 1

    @numba.njit
    def f2(x):
        return x[4]

    assert awkward1.to_list(f2(array)) == [1.1, 2.2, 3.3]
    assert counter[0] == 1

    assert awkward1.to_list(f2(array)) == [1.1, 2.2, 3.3]
    assert counter[0] == 1

    assert awkward1.to_list(f3(array)) == [[7.7, 8.8, 9.9], [6.6], [4.4, 5.5], [], [1.1, 2.2, 3.3]]

def test_indexedoptionarray():
    layout = awkward1.from_iter([[1.1, 2.2, 3.3], None, [], [4.4, 5.5], None, None, [6.6], [7.7, 8.8, 9.9]], highlevel=False)

    numbatype = awkward1._connect._numba.arrayview.tonumbatype(layout.form)
    assert awkward1_connect_numba_layout.typeof(layout).name == numbatype.name

    lookup1 = awkward1_connect_numba_arrayview.Lookup(layout)
    lookup2 = awkward1_connect_numba_arrayview.Lookup(layout.form)
    numbatype.form_fill(0, layout, lookup2)

    assert numpy.array_equal(lookup1.arrayptrs, lookup2.arrayptrs)
    assert numpy.array_equal(lookup1.sharedptrs == -1, lookup2.sharedptrs == -1)

    counter = [0]
    def materialize():
        counter[0] += 1
        return layout

    generator = awkward1.layout.ArrayGenerator(materialize, form=layout.form, length=len(layout))
    virtualarray = awkward1.layout.VirtualArray(generator)

    lookup3 = awkward1_connect_numba_arrayview.Lookup(virtualarray)
    assert len(lookup1.arrayptrs) + 3 == len(lookup3.arrayptrs)

    array = awkward1.Array(virtualarray)
    array.numba_type
    assert counter[0] == 0

    @numba.njit
    def f3(x):
        return x

    assert isinstance(f3(array).layout, awkward1.layout.VirtualArray)
    assert counter[0] == 0

    @numba.njit
    def f1(x):
        return x[3][1]

    assert f1(array) == 5.5
    assert counter[0] == 1

    assert f1(array) == 5.5
    assert counter[0] == 1

    @numba.njit
    def f2(x, i):
        return x[i]

    assert awkward1.to_list(f2(array, 3)) == [4.4, 5.5]
    assert counter[0] == 1

    assert awkward1.to_list(f2(array, 4)) == None
    assert counter[0] == 1

    assert awkward1.to_list(f2(array, 3)) == [4.4, 5.5]
    assert counter[0] == 1

    assert awkward1.to_list(f2(array, 4)) == None
    assert counter[0] == 1

    assert awkward1.to_list(f3(array)) == [[1.1, 2.2, 3.3], None, [], [4.4, 5.5], None, None, [6.6], [7.7, 8.8, 9.9]]

def test_bytemaskedarray():
    layout = awkward1.from_iter([[1.1, 2.2, 3.3], [999], [], [4.4, 5.5], [123, 321], [], [6.6], [7.7, 8.8, 9.9]], highlevel=False)
    layout = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array([False, True, False, False, True, True, False, False], dtype=numpy.int8)), layout, valid_when=False)

    numbatype = awkward1._connect._numba.arrayview.tonumbatype(layout.form)
    assert awkward1_connect_numba_layout.typeof(layout).name == numbatype.name

    lookup1 = awkward1_connect_numba_arrayview.Lookup(layout)
    lookup2 = awkward1_connect_numba_arrayview.Lookup(layout.form)
    numbatype.form_fill(0, layout, lookup2)

    assert numpy.array_equal(lookup1.arrayptrs, lookup2.arrayptrs)
    assert numpy.array_equal(lookup1.sharedptrs == -1, lookup2.sharedptrs == -1)

    counter = [0]
    def materialize():
        counter[0] += 1
        return layout

    generator = awkward1.layout.ArrayGenerator(materialize, form=layout.form, length=len(layout))
    virtualarray = awkward1.layout.VirtualArray(generator)

    lookup3 = awkward1_connect_numba_arrayview.Lookup(virtualarray)
    assert len(lookup1.arrayptrs) + 3 == len(lookup3.arrayptrs)

    array = awkward1.Array(virtualarray)
    array.numba_type
    assert counter[0] == 0

    @numba.njit
    def f3(x):
        return x

    assert isinstance(f3(array).layout, awkward1.layout.VirtualArray)
    assert counter[0] == 0

    @numba.njit
    def f1(x):
        return x[3][1]

    assert f1(array) == 5.5
    assert counter[0] == 1

    assert f1(array) == 5.5
    assert counter[0] == 1

    @numba.njit
    def f2(x, i):
        return x[i]

    assert awkward1.to_list(f2(array, 3)) == [4.4, 5.5]
    assert counter[0] == 1

    assert awkward1.to_list(f2(array, 4)) == None
    assert counter[0] == 1

    assert awkward1.to_list(f2(array, 3)) == [4.4, 5.5]
    assert counter[0] == 1

    assert awkward1.to_list(f2(array, 4)) == None
    assert counter[0] == 1

    assert awkward1.to_list(f3(array)) == [[1.1, 2.2, 3.3], None, [], [4.4, 5.5], None, None, [6.6], [7.7, 8.8, 9.9]]

def test_bitmaskedarray():
    layout = awkward1.from_iter([[1.1, 2.2, 3.3], [999], [], [4.4, 5.5], [123, 321], [], [6.6], [7.7, 8.8, 9.9], [3, 2, 1]], highlevel=False)
    layout = awkward1.layout.BitMaskedArray(awkward1.layout.IndexU8(numpy.packbits(numpy.array([False, True, False, False, True, True, False, False, False, True, True, True, True, True, True, True], dtype=numpy.bool))), layout, valid_when=False, length=9, lsb_order=False)

    numbatype = awkward1._connect._numba.arrayview.tonumbatype(layout.form)
    assert awkward1_connect_numba_layout.typeof(layout).name == numbatype.name

    lookup1 = awkward1_connect_numba_arrayview.Lookup(layout)
    lookup2 = awkward1_connect_numba_arrayview.Lookup(layout.form)
    numbatype.form_fill(0, layout, lookup2)

    assert numpy.array_equal(lookup1.arrayptrs, lookup2.arrayptrs)
    assert numpy.array_equal(lookup1.sharedptrs == -1, lookup2.sharedptrs == -1)

    counter = [0]
    def materialize():
        counter[0] += 1
        return layout

    generator = awkward1.layout.ArrayGenerator(materialize, form=layout.form, length=len(layout))
    virtualarray = awkward1.layout.VirtualArray(generator)

    lookup3 = awkward1_connect_numba_arrayview.Lookup(virtualarray)
    assert len(lookup1.arrayptrs) + 3 == len(lookup3.arrayptrs)

    array = awkward1.Array(virtualarray)
    array.numba_type
    assert counter[0] == 0

    @numba.njit
    def f3(x):
        return x

    assert isinstance(f3(array).layout, awkward1.layout.VirtualArray)
    assert counter[0] == 0

    @numba.njit
    def f1(x):
        return x[3][1]

    assert f1(array) == 5.5
    assert counter[0] == 1

    assert f1(array) == 5.5
    assert counter[0] == 1

    @numba.njit
    def f2(x, i):
        return x[i]

    assert awkward1.to_list(f2(array, 3)) == [4.4, 5.5]
    assert counter[0] == 1

    assert awkward1.to_list(f2(array, 4)) == None
    assert counter[0] == 1

    assert awkward1.to_list(f2(array, 3)) == [4.4, 5.5]
    assert counter[0] == 1

    assert awkward1.to_list(f2(array, 4)) == None
    assert counter[0] == 1

    assert awkward1.to_list(f3(array)) == [[1.1, 2.2, 3.3], None, [], [4.4, 5.5], None, None, [6.6], [7.7, 8.8, 9.9], [3.0, 2.0, 1.0]]

def test_unmaskedarray():
    layout = awkward1.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], highlevel=False)
    layout = awkward1.layout.UnmaskedArray(layout)

    numbatype = awkward1._connect._numba.arrayview.tonumbatype(layout.form)
    assert awkward1_connect_numba_layout.typeof(layout).name == numbatype.name

    lookup1 = awkward1_connect_numba_arrayview.Lookup(layout)
    lookup2 = awkward1_connect_numba_arrayview.Lookup(layout.form)
    numbatype.form_fill(0, layout, lookup2)

    assert numpy.array_equal(lookup1.arrayptrs, lookup2.arrayptrs)
    assert numpy.array_equal(lookup1.sharedptrs == -1, lookup2.sharedptrs == -1)

    counter = [0]
    def materialize():
        counter[0] += 1
        return layout

    generator = awkward1.layout.ArrayGenerator(materialize, form=layout.form, length=len(layout))
    virtualarray = awkward1.layout.VirtualArray(generator)

    lookup3 = awkward1_connect_numba_arrayview.Lookup(virtualarray)
    assert len(lookup1.arrayptrs) + 3 == len(lookup3.arrayptrs)

    array = awkward1.Array(virtualarray)
    array.numba_type
    assert counter[0] == 0

    @numba.njit
    def f3(x):
        return x

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

def test_recordarray():
    layout = awkward1.from_iter([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}], highlevel=False)

    numbatype = awkward1._connect._numba.arrayview.tonumbatype(layout.form)
    assert awkward1_connect_numba_layout.typeof(layout).name == numbatype.name

    lookup1 = awkward1_connect_numba_arrayview.Lookup(layout)
    lookup2 = awkward1_connect_numba_arrayview.Lookup(layout.form)
    numbatype.form_fill(0, layout, lookup2)

    assert numpy.array_equal(lookup1.arrayptrs, lookup2.arrayptrs)
    assert numpy.array_equal(lookup1.sharedptrs == -1, lookup2.sharedptrs == -1)

    counter = [0]
    def materialize():
        counter[0] += 1
        return layout

    generator = awkward1.layout.ArrayGenerator(materialize, form=layout.form, length=len(layout))
    virtualarray = awkward1.layout.VirtualArray(generator)

    lookup3 = awkward1_connect_numba_arrayview.Lookup(virtualarray)
    assert len(lookup1.arrayptrs) + 3 == len(lookup3.arrayptrs)

    array = awkward1.Array(virtualarray)
    array.numba_type
    assert counter[0] == 0

    @numba.njit
    def f3(x):
        return x

    assert isinstance(f3(array).layout, awkward1.layout.VirtualArray)
    assert counter[0] == 0

    @numba.njit
    def f1a(x):
        return x["x"][2]

    assert f1a(array) == 2.2
    assert counter[0] == 1

    assert f1a(array) == 2.2
    assert counter[0] == 1

    @numba.njit
    def f1b(x):
        return x[2]["x"]

    assert f1b(array) == 2.2
    assert counter[0] == 1

    assert f1b(array) == 2.2
    assert counter[0] == 1

    @numba.njit
    def f1c(x):
        return x.x[2]

    assert f1c(array) == 2.2
    assert counter[0] == 1

    assert f1c(array) == 2.2
    assert counter[0] == 1

    @numba.njit
    def f1d(x):
        return x[2].x

    assert f1d(array) == 2.2
    assert counter[0] == 1

    assert f1d(array) == 2.2
    assert counter[0] == 1

    @numba.njit
    def f2a(x):
        return x["y"][2]

    assert awkward1.to_list(f2a(array)) == [2, 2]
    assert counter[0] == 1

    assert awkward1.to_list(f2a(array)) == [2, 2]
    assert counter[0] == 1

    @numba.njit
    def f2b(x):
        return x[2]["y"]

    assert awkward1.to_list(f2b(array)) == [2, 2]
    assert counter[0] == 1

    assert awkward1.to_list(f2b(array)) == [2, 2]
    assert counter[0] == 1

    @numba.njit
    def f2c(x):
        return x.y[2]

    assert awkward1.to_list(f2a(array)) == [2, 2]
    assert counter[0] == 1

    assert awkward1.to_list(f2a(array)) == [2, 2]
    assert counter[0] == 1

    @numba.njit
    def f2d(x):
        return x[2].y

    assert awkward1.to_list(f2b(array)) == [2, 2]
    assert counter[0] == 1

    assert awkward1.to_list(f2b(array)) == [2, 2]
    assert counter[0] == 1

    assert awkward1.to_list(f3(array)) == [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}]

def test_tuplearray():
    layout = awkward1.from_iter([(0.0, []), (1.1, [1]), (2.2, [2, 2]), (3.3, [3, 3, 3]), (4.4, [4, 4, 4, 4])], highlevel=False)

    numbatype = awkward1._connect._numba.arrayview.tonumbatype(layout.form)
    assert awkward1_connect_numba_layout.typeof(layout).name == numbatype.name

    lookup1 = awkward1_connect_numba_arrayview.Lookup(layout)
    lookup2 = awkward1_connect_numba_arrayview.Lookup(layout.form)
    numbatype.form_fill(0, layout, lookup2)

    assert numpy.array_equal(lookup1.arrayptrs, lookup2.arrayptrs)
    assert numpy.array_equal(lookup1.sharedptrs == -1, lookup2.sharedptrs == -1)

    counter = [0]
    def materialize():
        counter[0] += 1
        return layout

    generator = awkward1.layout.ArrayGenerator(materialize, form=layout.form, length=len(layout))
    virtualarray = awkward1.layout.VirtualArray(generator)

    lookup3 = awkward1_connect_numba_arrayview.Lookup(virtualarray)
    assert len(lookup1.arrayptrs) + 3 == len(lookup3.arrayptrs)

    array = awkward1.Array(virtualarray)
    array.numba_type
    assert counter[0] == 0

    @numba.njit
    def f3(x):
        return x

    assert isinstance(f3(array).layout, awkward1.layout.VirtualArray)
    assert counter[0] == 0

    @numba.njit
    def f1a(x):
        return x["0"][2]

    assert f1a(array) == 2.2
    assert counter[0] == 1

    assert f1a(array) == 2.2
    assert counter[0] == 1

    @numba.njit
    def f1b(x):
        return x[2]["0"]

    assert f1b(array) == 2.2
    assert counter[0] == 1

    assert f1b(array) == 2.2
    assert counter[0] == 1

    @numba.njit
    def f2a(x):
        return x["1"][2]

    assert awkward1.to_list(f2a(array)) == [2, 2]
    assert counter[0] == 1

    assert awkward1.to_list(f2a(array)) == [2, 2]
    assert counter[0] == 1

    @numba.njit
    def f2b(x):
        return x[2]["1"]

    assert awkward1.to_list(f2b(array)) == [2, 2]
    assert counter[0] == 1

    assert awkward1.to_list(f2b(array)) == [2, 2]
    assert counter[0] == 1

    assert awkward1.to_list(f3(array)) == [(0.0, []), (1.1, [1]), (2.2, [2, 2]), (3.3, [3, 3, 3]), (4.4, [4, 4, 4, 4])]

def test_unionarray():
    layout = awkward1.from_iter([0.0, [], 1.1, [1], 2.2, [2, 2], 3.3, [3, 3, 3], 4.4, [4, 4, 4, 4]], highlevel=False)

    numbatype = awkward1._connect._numba.arrayview.tonumbatype(layout.form)
    assert awkward1_connect_numba_layout.typeof(layout).name == numbatype.name

    lookup1 = awkward1_connect_numba_arrayview.Lookup(layout)
    lookup2 = awkward1_connect_numba_arrayview.Lookup(layout.form)
    numbatype.form_fill(0, layout, lookup2)

    assert numpy.array_equal(lookup1.arrayptrs, lookup2.arrayptrs)
    assert numpy.array_equal(lookup1.sharedptrs == -1, lookup2.sharedptrs == -1)

    counter = [0]
    def materialize():
        counter[0] += 1
        return layout

    generator = awkward1.layout.ArrayGenerator(materialize, form=layout.form, length=len(layout))
    virtualarray = awkward1.layout.VirtualArray(generator)

    lookup3 = awkward1_connect_numba_arrayview.Lookup(virtualarray)
    assert len(lookup1.arrayptrs) + 3 == len(lookup3.arrayptrs)

    array = awkward1.Array(virtualarray)
    array.numba_type
    assert counter[0] == 0

    @numba.njit
    def f3(x):
        return x

    assert isinstance(f3(array).layout, awkward1.layout.VirtualArray)
    assert counter[0] == 0

    assert awkward1.to_list(f3(array)) == [0.0, [], 1.1, [1], 2.2, [2, 2], 3.3, [3, 3, 3], 4.4, [4, 4, 4, 4]]

def test_deep_virtualarrays():
    one = awkward1.from_iter([0.0, 1.1, 2.2, 3.3, 4.4], highlevel=False)
    two = awkward1.from_iter([[], [1], [2, 2], [3, 3, 3], [4, 4, 4, 4]], highlevel=False)

    counter = [0, 0]
    def materialize1():
        counter[0] += 1
        return one

    generator1 = awkward1.layout.ArrayGenerator(materialize1, form=one.form, length=len(one))
    vone = awkward1.layout.VirtualArray(generator1)

    def materialize2():
        counter[1] += 1
        return two

    generator2 = awkward1.layout.ArrayGenerator(materialize2, form=two.form, length=len(two))
    vtwo = awkward1.layout.VirtualArray(generator2)

    recordarray = awkward1.layout.RecordArray([vone, vtwo], ["x", "y"])
    array = awkward1.Array(recordarray)
    array.numba_type
    assert counter == [0, 0]

    @numba.njit
    def f3(x):
        return x

    tmp1 = f3(array).layout
    assert isinstance(tmp1, awkward1.layout.RecordArray)
    assert isinstance(tmp1.field(0), awkward1.layout.VirtualArray)
    assert isinstance(tmp1.field(1), awkward1.layout.VirtualArray)
    assert counter == [0, 0]

    @numba.njit
    def f1a(x):
        return x["x"]

    tmp2a = f1a(array).layout
    assert isinstance(tmp2a, awkward1.layout.VirtualArray)
    assert counter == [0, 0]

    @numba.njit
    def f1b(x):
        return x["x"][2]

    tmp2b = f1b(array)
    assert tmp2b == 2.2
    assert counter == [1, 0]

    @numba.njit
    def f2a(x):
        return x["y"]

    tmp3a = f2a(array).layout
    assert isinstance(tmp3a, awkward1.layout.VirtualArray)
    assert counter == [1, 0]

    @numba.njit
    def f2b(x):
        return x["y"][2]

    tmp3b = f2b(array).layout
    assert isinstance(tmp3b, awkward1.layout.NumpyArray)
    assert awkward1.to_list(tmp3b) == [2, 2]
    assert counter == [1, 1]

def test_nested_virtualness():
    counter = [0, 0]

    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))

    def materialize1():
        counter[1] += 1
        return content

    generator1 = awkward1.layout.ArrayGenerator(materialize1, form=content.form, length=len(content))
    virtual1 = awkward1.layout.VirtualArray(generator1)

    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10], dtype=numpy.int64))
    listarray = awkward1.layout.ListOffsetArray64(offsets, virtual1)

    def materialize2():
        counter[0] += 1
        return listarray

    generator2 = awkward1.layout.ArrayGenerator(materialize2, form=listarray.form, length=len(listarray))
    virtual2 = awkward1.layout.VirtualArray(generator2)
    array = awkward1.Array(virtual2)

    assert counter == [0, 0]

    @numba.njit
    def f1(x, i):
        return x[i]

    tmp1 = f1(array, 2)
    assert counter == [1, 0]

    tmp2 = f1(tmp1, 1)
    assert tmp2 == 4.4
    assert counter == [1, 1]
