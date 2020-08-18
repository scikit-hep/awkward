# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import
from collections import defaultdict

import awkward1
import numba


def test_virtual_slicearray_numba():
    materialize_count = defaultdict(int)

    def gen(x, name):
        materialize_count[name] += 1
        return x

    array = awkward1.Array([1, 2, 3, 4, 5])
    virtual = awkward1.virtual(lambda: gen(array, "array"), length=len(array), form=array.layout.form)
    slicearray = awkward1.Array([True, False, False, True, False])
    slicedvirtual = virtual[slicearray]

    @numba.njit
    def dostuff(array):
        x = 0
        for item in array:
            x += item
        return x

    assert slicedvirtual.layout.form.form is None
    assert materialize_count["array"] == 0
    assert dostuff(slicedvirtual) == 5
    assert materialize_count["array"] == 2
    materialize_count.clear()

    x = awkward1.Array([1, 2, 3, 4, 5])
    y = x * 10.
    xv = awkward1.virtual(lambda: gen(x, "x"), length=len(x), form=x.layout.form)
    yv = awkward1.virtual(lambda: gen(y, "y"), length=len(y), form=y.layout.form)
    array = awkward1.zip({"x": xv, "y": yv})
    virtual = awkward1.virtual(lambda: gen(array, "array"), length=len(array), form=array.layout.form)
    slicedvirtual = virtual[slicearray]

    @numba.njit
    def dostuff(array):
        x = 0
        for item in array:
            x += item.x
        return x

    assert materialize_count["x"] == 0
    assert materialize_count["y"] == 0
    assert materialize_count["array"] == 0
    assert dostuff(slicedvirtual) == 5
    assert materialize_count["x"] == 1
    assert materialize_count["y"] == 0
    assert materialize_count["array"] == 2
    materialize_count.clear()


def test_virtual_slicefield_numba():
    materialize_count = defaultdict(int)

    def gen(x, name):
        materialize_count[name] += 1
        return x

    x = awkward1.Array([1, 2, 3, 4, 5])
    y = x * 10.
    xv = awkward1.virtual(lambda: gen(x, "x"), length=len(x), form=x.layout.form)
    yv = awkward1.virtual(lambda: gen(y, "y"), length=len(y), form=y.layout.form)
    array = awkward1.zip({"x": xv, "y": yv})
    virtual = awkward1.virtual(lambda: gen(array, "array"), length=len(array), form=array.layout.form)
    slicedvirtual = virtual.x

    @numba.njit
    def dostuff(array):
        x = 0
        for item in array:
            x += item
        return x

    assert materialize_count["x"] == 0
    assert materialize_count["y"] == 0
    assert materialize_count["array"] == 0
    assert dostuff(slicedvirtual) == 15
    assert materialize_count["x"] == 1
    assert materialize_count["y"] == 0
    assert materialize_count["array"] == 2
