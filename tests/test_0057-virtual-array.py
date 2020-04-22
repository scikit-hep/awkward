# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import pytest
import numpy

import awkward1

def test_forms():
    form = awkward1.forms.NumpyForm([], 8, "d")
    assert form == form
    assert form.inner_shape == []
    assert form.itemsize == 8
    assert form.primitive == "float64"
    assert form.has_identities == False
    assert form.parameters == {}
    assert json.loads(form.tojson(False, True)) == {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}}
    assert json.loads(str(form)) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}

    form = awkward1.forms.NumpyForm([1, 2, 3], 8, "d", has_identities=True, parameters={"hey": ["you", {"there": 3}]})
    assert form == form
    assert form.inner_shape == [1, 2, 3]
    assert form.itemsize == 8
    assert form.primitive == "float64"
    assert form.has_identities == True
    assert form.parameters == {"hey": ["you", {"there": 3}]}
    assert form.parameter("hey") == ["you", {"there": 3}]
    assert json.loads(form.tojson(False, True)) == {"class": "NumpyArray", "inner_shape": [1, 2, 3], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": True, "parameters": {"hey": ["you", {"there": 3}]}}
    assert json.loads(str(form)) == {"class": "NumpyArray", "inner_shape": [1, 2, 3], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": True, "parameters": {"hey": ["you", {"there": 3}]}}

    form = awkward1.forms.BitMaskedForm("i8", awkward1.forms.NumpyForm([], 8, "d"), True, False, parameters={"hey": ["you"]})
    assert form == form
    assert json.loads(form.tojson(False, True)) == {"class": "BitMaskedArray", "mask": "i8", "content": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}}, "valid_when": True, "lsb_order": False, "has_identities": False, "parameters": {"hey": ["you"]}}
    assert json.loads(str(form)) == {"class": "BitMaskedArray", "mask": "i8", "content": "float64", "valid_when": True, "lsb_order": False, "parameters": {"hey": ["you"]}}
    assert form.mask == "i8"
    assert json.loads(str(form.content)) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert form.valid_when == True
    assert form.lsb_order == False
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]

    form = awkward1.forms.ByteMaskedForm("i8", awkward1.forms.NumpyForm([], 8, "d"), True, parameters={"hey": ["you"]})
    assert form == form
    assert json.loads(form.tojson(False, True)) == {"class": "ByteMaskedArray", "mask": "i8", "content": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}}, "valid_when": True, "has_identities": False, "parameters": {"hey": ["you"]}}
    assert json.loads(str(form)) == {"class": "ByteMaskedArray", "mask": "i8", "content": "float64", "valid_when": True, "parameters": {"hey": ["you"]}}
    assert form.mask == "i8"
    assert json.loads(str(form.content)) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert form.valid_when == True
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]

    form = awkward1.forms.EmptyForm(parameters={"hey": ["you"]})
    assert form == form
    assert json.loads(form.tojson(False, True)) == {"class": "EmptyArray", "has_identities": False, "parameters": {"hey": ["you"]}}
    assert json.loads(str(form)) == {"class": "EmptyArray", "parameters": {"hey": ["you"]}}
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]

    form = awkward1.forms.IndexedForm("i64", awkward1.forms.NumpyForm([], 8, "d"), parameters={"hey": ["you"]})
    assert form == form
    assert json.loads(form.tojson(False, True)) == {"class": "IndexedArray64", "index": "i64", "content": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}}, "has_identities": False, "parameters": {"hey": ["you"]}}
    assert json.loads(str(form)) == {"class": "IndexedArray64", "index": "i64", "content": "float64", "parameters": {"hey": ["you"]}}
    assert form.index == "i64"
    assert json.loads(str(form.content)) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]

    form = awkward1.forms.IndexedOptionForm("i64", awkward1.forms.NumpyForm([], 8, "d"), parameters={"hey": ["you"]})
    assert form == form
    assert json.loads(form.tojson(False, True)) == {"class": "IndexedOptionArray64", "index": "i64", "content": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}}, "has_identities": False, "parameters": {"hey": ["you"]}}
    assert json.loads(str(form)) == {"class": "IndexedOptionArray64", "index": "i64", "content": "float64", "parameters": {"hey": ["you"]}}
    assert form.index == "i64"
    assert json.loads(str(form.content)) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]

    form = awkward1.forms.ListForm("i64", "i64", awkward1.forms.NumpyForm([], 8, "d"), parameters={"hey": ["you"]})
    assert form == form
    assert json.loads(form.tojson(False, True)) == {"class": "ListArray64", "starts": "i64", "stops": "i64", "content": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}}, "has_identities": False, "parameters": {"hey": ["you"]}}
    assert json.loads(str(form)) == {"class": "ListArray64", "starts": "i64", "stops": "i64", "content": "float64", "parameters": {"hey": ["you"]}}
    assert form.starts == "i64"
    assert form.stops == "i64"
    assert json.loads(str(form.content)) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]

    form = awkward1.forms.ListOffsetForm("i64", awkward1.forms.NumpyForm([], 8, "d"), parameters={"hey": ["you"]})
    assert form == form
    assert json.loads(form.tojson(False, True)) == {"class": "ListOffsetArray64", "offsets": "i64", "content": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}}, "has_identities": False, "parameters": {"hey": ["you"]}}
    assert json.loads(str(form)) == {"class": "ListOffsetArray64", "offsets": "i64", "content": "float64", "parameters": {"hey": ["you"]}}
    assert form.offsets == "i64"
    assert json.loads(str(form.content)) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]

    form = awkward1.forms.RecordForm({"one": awkward1.forms.NumpyForm([], 8, "d"), "two": awkward1.forms.NumpyForm([], 1, "?")}, parameters={"hey": ["you"]})
    assert form == form
    assert json.loads(form.tojson(False, True)) == {"class": "RecordArray", "contents": {"one": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}}, "two": {"class": "NumpyArray", "inner_shape": [], "itemsize": 1, "format": "?", "primitive": "bool", "has_identities": False, "parameters": {}}}, "has_identities": False, "parameters": {"hey": ["you"]}}
    assert json.loads(str(form)) == {"class": "RecordArray", "contents": {"one": "float64", "two": "bool"}, "parameters": {"hey": ["you"]}}
    if not awkward1._util.py27 and not awkward1._util.py35:
        assert [json.loads(str(x)) for x in form.values()] == [{"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}, {"class": "NumpyArray", "itemsize": 1, "format": "?", "primitive": "bool"}]
        assert {n: json.loads(str(x)) for n, x in form.contents.items()} == {"one": {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}, "two": {"class": "NumpyArray", "itemsize": 1, "format": "?", "primitive": "bool"}}
    assert json.loads(str(form.content("one"))) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert json.loads(str(form.content("two"))) == {"class": "NumpyArray", "itemsize": 1, "format": "?", "primitive": "bool"}
    if not awkward1._util.py27 and not awkward1._util.py35:
        assert json.loads(str(form.content(0))) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
        assert json.loads(str(form.content(1))) == {"class": "NumpyArray", "itemsize": 1, "format": "?", "primitive": "bool"}
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]

    form = awkward1.forms.RecordForm([awkward1.forms.NumpyForm([], 8, "d"), awkward1.forms.NumpyForm([], 1, "?")], parameters={"hey": ["you"]})
    assert form == form
    assert json.loads(form.tojson(False, True)) == {"class": "RecordArray", "contents": [{"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}}, {"class": "NumpyArray", "inner_shape": [], "itemsize": 1, "format": "?", "primitive": "bool", "has_identities": False, "parameters": {}}], "has_identities": False, "parameters": {"hey": ["you"]}}
    assert json.loads(str(form)) == {"class": "RecordArray", "contents": ["float64", "bool"], "parameters": {"hey": ["you"]}}
    assert [json.loads(str(x)) for x in form.values()] == [{"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}, {"class": "NumpyArray", "itemsize": 1, "format": "?", "primitive": "bool"}]
    assert {n: json.loads(str(x)) for n, x in form.contents.items()} == {"0": {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}, "1": {"class": "NumpyArray", "itemsize": 1, "format": "?", "primitive": "bool"}}
    assert json.loads(str(form.content(0))) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert json.loads(str(form.content(1))) == {"class": "NumpyArray", "itemsize": 1, "format": "?", "primitive": "bool"}
    assert json.loads(str(form.content("0"))) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert json.loads(str(form.content("1"))) == {"class": "NumpyArray", "itemsize": 1, "format": "?", "primitive": "bool"}
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]

    form = awkward1.forms.RegularForm(awkward1.forms.NumpyForm([], 8, "d"), 10, parameters={"hey": ["you"]})
    assert form == form
    assert json.loads(form.tojson(False, True)) == {"class": "RegularArray", "content": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}}, "size": 10, "has_identities": False, "parameters": {"hey": ["you"]}}
    assert json.loads(str(form)) == {"class": "RegularArray", "content": "float64", "size": 10, "parameters": {"hey": ["you"]}}
    assert json.loads(str(form.content)) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert form.size == 10
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]

    form = awkward1.forms.UnionForm("i8", "i64", [awkward1.forms.NumpyForm([], 8, "d"), awkward1.forms.NumpyForm([], 1, "?")], parameters={"hey": ["you"]})
    assert form == form
    assert json.loads(form.tojson(False, True)) == {"class": "UnionArray8_64", "tags": "i8", "index": "i64", "contents": [{"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}}, {"class": "NumpyArray", "inner_shape": [], "itemsize": 1, "format": "?", "primitive": "bool", "has_identities": False, "parameters": {}}], "has_identities": False, "parameters": {"hey": ["you"]}}
    assert json.loads(str(form)) == {"class": "UnionArray8_64", "tags": "i8", "index": "i64", "contents": ["float64", "bool"], "parameters": {"hey": ["you"]}}
    assert form.tags == "i8"
    assert form.index == "i64"
    assert json.loads(str(form.contents)) == [{"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}, {"class": "NumpyArray", "itemsize": 1, "format": "?", "primitive": "bool"}]
    assert json.loads(str(form.content(0))) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert json.loads(str(form.content(1))) == {"class": "NumpyArray", "itemsize": 1, "format": "?", "primitive": "bool"}
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]

    form = awkward1.forms.UnmaskedForm(awkward1.forms.NumpyForm([], 8, "d"), parameters={"hey": ["you"]})
    assert form == form
    assert json.loads(form.tojson(False, True)) == {"class": "UnmaskedArray", "content": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}}, "has_identities": False, "parameters": {"hey": ["you"]}}
    assert json.loads(str(form)) == {"class": "UnmaskedArray", "content": "float64", "parameters": {"hey": ["you"]}}
    assert json.loads(str(form.content)) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]

def fcn():
    return awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))

def test_basic():
    generator = awkward1.virtual.ArrayGenerator(fcn, form=awkward1.forms.NumpyForm([], 8, "d"), length=5)

    d = {}
    cache = awkward1.virtual.ArrayCache(d)

    virtualarray = awkward1.layout.VirtualArray(generator, cache)
    assert virtualarray.peek_array is None
    assert virtualarray.array is not None
    assert awkward1.to_list(virtualarray.peek_array) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.to_list(virtualarray.array) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.to_list(d[virtualarray.cache_key]) == [1.1, 2.2, 3.3, 4.4, 5.5]
