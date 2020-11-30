# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import collections
import json

import pytest
import numpy as np
import awkward1 as ak


def test_virtual_record():
    materialize_count = collections.defaultdict(int)

    def gen(x, name):
        materialize_count[name] += 1
        return x

    x1 = ak.Array([1, 2, 3, 4, 5])
    x2 = ak.Array([1, 2, 3, 4, 5])
    x = ak.zip({"x1": x1, "x2": x2}, with_name="xthing")
    assert x.layout.purelist_parameter("__record__") == "xthing"
    xv = ak.virtual(lambda: gen(x, "x"), length=len(x), form=x.layout.form)
    assert xv.layout.purelist_parameter("__record__") == "xthing"
    y = x1 * 10.0
    yv = ak.virtual(lambda: gen(y, "y"), length=len(y), form=y.layout.form)
    array = ak.zip({"x": xv, "y": yv}, with_name="Point", depth_limit=1)
    assert array.layout.purelist_parameter("__record__") == "Point"
    virtual = ak.virtual(
        lambda: gen(array, "array"), length=len(array), form=array.layout.form
    )
    assert virtual.layout.purelist_parameter("__record__") == "Point"
    assert len(materialize_count) == 0

    slicedvirtual = virtual.x
    assert len(materialize_count) == 0
    assert slicedvirtual.layout.purelist_parameter("__record__") == "xthing"
    assert len(materialize_count) == 0

    slicedvirtual = virtual[["x", "y"]]
    assert len(materialize_count) == 0
    assert slicedvirtual.layout.purelist_parameter("__record__") == None
    assert len(materialize_count) == 0

    slicedvirtual = virtual[::2]
    assert len(materialize_count) == 0
    assert slicedvirtual.layout.purelist_parameter("__record__") == "Point"
    assert len(materialize_count) == 0

    slicedvirtual = virtual[:3]
    assert len(materialize_count) == 0
    assert slicedvirtual.layout.purelist_parameter("__record__") == "Point"
    assert len(materialize_count) == 0

    slicedvirtual = virtual[ak.Array([True, False, False, True, False])]
    assert len(materialize_count) == 0
    assert slicedvirtual.layout.purelist_parameter("__record__") == "Point"
    assert len(materialize_count) == 0


def test_virtual_slice_numba():
    numba = pytest.importorskip("numba")
    materialize_count = collections.defaultdict(int)

    def gen(x, name):
        materialize_count[name] += 1
        return x

    x1 = ak.Array([1, 2, 3, 4, 5])
    x2 = ak.Array([1, 2, 3, 4, 5])
    x = ak.zip({"x1": x1, "x2": x2}, with_name="xthing")
    xv = ak.virtual(lambda: gen(x, "x"), length=len(x), form=x.layout.form)
    y = x1 * 10.0
    yv = ak.virtual(lambda: gen(y, "y"), length=len(y), form=y.layout.form)
    array = ak.zip({"x": xv, "y": yv}, with_name="Point", depth_limit=1)
    virtual = ak.virtual(
        lambda: gen(array, "array"),
        length=len(array),
        form=ak.forms.Form.fromjson(
            json.dumps(
                {
                    "class": "RecordArray",
                    "contents": {
                        "x": {
                            "class": "VirtualArray",
                            "form": json.loads(str(x.layout.form)),
                            "has_length": True,
                        },
                        "y": {
                            "class": "VirtualArray",
                            "form": json.loads(str(y.layout.form)),
                            "has_length": True,
                        },
                    },
                    "parameters": {"__record__": "Point"},
                }
            )
        ),
    )

    @numba.njit
    def dostuff(array):
        x = 0
        for item in array:
            x += item
        return x

    assert dostuff(virtual.x.x1) == 15
    assert dict(materialize_count) == {"x": 1, "array": 1}
    materialize_count.clear()

    @numba.njit
    def dostuff(array):
        x = 0
        for item in array:
            x += item.x.x1
        return x

    assert dostuff(virtual[["x"]]) == 15
    assert dict(materialize_count) == {"x": 1, "array": 2}
    materialize_count.clear()

    assert dostuff(virtual[::2]) == 9
    assert dict(materialize_count) == {"x": 1, "array": 2}
    materialize_count.clear()

    assert dostuff(virtual[:3]) == 6
    assert dict(materialize_count) == {"x": 1, "array": 1}
    materialize_count.clear()

    slicedvirtual = virtual[ak.Array([True, False, False, True, False])]
    assert dostuff(slicedvirtual) == 5
    assert dict(materialize_count) == {"x": 1, "array": 2}
    materialize_count.clear()
