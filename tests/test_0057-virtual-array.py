# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json
try:
    # pybind11 only supports cPickle protocol 2+ (-1 in pickle.dumps)
    # (automatically satisfied in Python 3; this is just to keep testing Python 2.7)
    import cPickle as pickle
except ImportError:
    import pickle

import pytest
import numpy

import awkward1

def test_forms():
    form = awkward1.forms.NumpyForm([], 8, "d")
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, False)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, True)) == form
    assert form.inner_shape == []
    assert form.itemsize == 8
    assert form.primitive == "float64"
    assert form.has_identities == False
    assert form.parameters == {}
    assert form.form_key is None
    assert json.loads(form.tojson(False, True)) == {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}, "form_key": None}
    assert json.loads(str(form)) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}

    form = awkward1.forms.NumpyForm([1, 2, 3], 8, "d", has_identities=True, parameters={"hey": ["you", {"there": 3}]}, form_key="yowzers")
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, False)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, True)) == form
    assert form.inner_shape == [1, 2, 3]
    assert form.itemsize == 8
    assert form.primitive == "float64"
    assert form.has_identities == True
    assert form.parameters == {"hey": ["you", {"there": 3}]}
    assert form.parameter("hey") == ["you", {"there": 3}]
    assert form.form_key == "yowzers"
    assert json.loads(form.tojson(False, True)) == {"class": "NumpyArray", "inner_shape": [1, 2, 3], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": True, "parameters": {"hey": ["you", {"there": 3}]}, "form_key": "yowzers"}
    assert json.loads(str(form)) == {"class": "NumpyArray", "inner_shape": [1, 2, 3], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": True, "parameters": {"hey": ["you", {"there": 3}]}, "form_key": "yowzers"}

    form = awkward1.forms.BitMaskedForm("i8", awkward1.forms.NumpyForm([], 8, "d"), True, False, parameters={"hey": ["you"]}, form_key="yowzers")
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, False)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, True)) == form
    assert json.loads(form.tojson(False, True)) == {"class": "BitMaskedArray", "mask": "i8", "content": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}, "form_key": None}, "valid_when": True, "lsb_order": False, "has_identities": False, "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert json.loads(str(form)) == {"class": "BitMaskedArray", "mask": "i8", "content": "float64", "valid_when": True, "lsb_order": False, "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert form.mask == "i8"
    assert json.loads(str(form.content)) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert form.valid_when == True
    assert form.lsb_order == False
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]
    assert form.form_key == "yowzers"

    form = awkward1.forms.ByteMaskedForm("i8", awkward1.forms.NumpyForm([], 8, "d"), True, parameters={"hey": ["you"]}, form_key="yowzers")
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, False)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, True)) == form
    assert json.loads(form.tojson(False, True)) == {"class": "ByteMaskedArray", "mask": "i8", "content": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}, "form_key": None}, "valid_when": True, "has_identities": False, "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert json.loads(str(form)) == {"class": "ByteMaskedArray", "mask": "i8", "content": "float64", "valid_when": True, "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert form.mask == "i8"
    assert json.loads(str(form.content)) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert form.valid_when == True
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]
    assert form.form_key == "yowzers"

    form = awkward1.forms.EmptyForm(parameters={"hey": ["you"]}, form_key="yowzers")
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, False)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, True)) == form
    assert json.loads(form.tojson(False, True)) == {"class": "EmptyArray", "has_identities": False, "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert json.loads(str(form)) == {"class": "EmptyArray", "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]
    assert form.form_key == "yowzers"

    form = awkward1.forms.IndexedForm("i64", awkward1.forms.NumpyForm([], 8, "d"), parameters={"hey": ["you"]}, form_key="yowzers")
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, False)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, True)) == form
    assert json.loads(form.tojson(False, True)) == {"class": "IndexedArray64", "index": "i64", "content": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}, "form_key": None}, "has_identities": False, "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert json.loads(str(form)) == {"class": "IndexedArray64", "index": "i64", "content": "float64", "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert form.index == "i64"
    assert json.loads(str(form.content)) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]
    assert form.form_key == "yowzers"

    form = awkward1.forms.IndexedOptionForm("i64", awkward1.forms.NumpyForm([], 8, "d"), parameters={"hey": ["you"]}, form_key="yowzers")
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, False)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, True)) == form
    assert json.loads(form.tojson(False, True)) == {"class": "IndexedOptionArray64", "index": "i64", "content": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}, "form_key": None}, "has_identities": False, "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert json.loads(str(form)) == {"class": "IndexedOptionArray64", "index": "i64", "content": "float64", "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert form.index == "i64"
    assert json.loads(str(form.content)) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]
    assert form.form_key == "yowzers"

    form = awkward1.forms.ListForm("i64", "i64", awkward1.forms.NumpyForm([], 8, "d"), parameters={"hey": ["you"]}, form_key="yowzers")
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, False)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, True)) == form
    assert json.loads(form.tojson(False, True)) == {"class": "ListArray64", "starts": "i64", "stops": "i64", "content": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}, "form_key": None}, "has_identities": False, "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert json.loads(str(form)) == {"class": "ListArray64", "starts": "i64", "stops": "i64", "content": "float64", "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert form.starts == "i64"
    assert form.stops == "i64"
    assert json.loads(str(form.content)) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]
    assert form.form_key == "yowzers"

    form = awkward1.forms.ListOffsetForm("i64", awkward1.forms.NumpyForm([], 8, "d"), parameters={"hey": ["you"]}, form_key="yowzers")
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, False)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, True)) == form
    assert json.loads(form.tojson(False, True)) == {"class": "ListOffsetArray64", "offsets": "i64", "content": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}, "form_key": None}, "has_identities": False, "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert json.loads(str(form)) == {"class": "ListOffsetArray64", "offsets": "i64", "content": "float64", "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert form.offsets == "i64"
    assert json.loads(str(form.content)) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]
    assert form.form_key == "yowzers"

    form = awkward1.forms.RecordForm({"one": awkward1.forms.NumpyForm([], 8, "d"), "two": awkward1.forms.NumpyForm([], 1, "?")}, parameters={"hey": ["you"]}, form_key="yowzers")
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, False)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, True)) == form
    assert json.loads(form.tojson(False, True)) == {"class": "RecordArray", "contents": {"one": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}, "form_key": None}, "two": {"class": "NumpyArray", "inner_shape": [], "itemsize": 1, "format": "?", "primitive": "bool", "has_identities": False, "parameters": {}, "form_key": None}}, "has_identities": False, "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert json.loads(str(form)) == {"class": "RecordArray", "contents": {"one": "float64", "two": "bool"}, "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
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
    assert form.form_key == "yowzers"

    form = awkward1.forms.RecordForm([awkward1.forms.NumpyForm([], 8, "d"), awkward1.forms.NumpyForm([], 1, "?")], parameters={"hey": ["you"]}, form_key="yowzers")
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, False)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, True)) == form
    assert json.loads(form.tojson(False, True)) == {"class": "RecordArray", "contents": [{"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}, "form_key": None}, {"class": "NumpyArray", "inner_shape": [], "itemsize": 1, "format": "?", "primitive": "bool", "has_identities": False, "parameters": {}, "form_key": None}], "has_identities": False, "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert json.loads(str(form)) == {"class": "RecordArray", "contents": ["float64", "bool"], "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert [json.loads(str(x)) for x in form.values()] == [{"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}, {"class": "NumpyArray", "itemsize": 1, "format": "?", "primitive": "bool"}]
    assert {n: json.loads(str(x)) for n, x in form.contents.items()} == {"0": {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}, "1": {"class": "NumpyArray", "itemsize": 1, "format": "?", "primitive": "bool"}}
    assert json.loads(str(form.content(0))) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert json.loads(str(form.content(1))) == {"class": "NumpyArray", "itemsize": 1, "format": "?", "primitive": "bool"}
    assert json.loads(str(form.content("0"))) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert json.loads(str(form.content("1"))) == {"class": "NumpyArray", "itemsize": 1, "format": "?", "primitive": "bool"}
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]
    assert form.form_key == "yowzers"

    form = awkward1.forms.RegularForm(awkward1.forms.NumpyForm([], 8, "d"), 10, parameters={"hey": ["you"]}, form_key="yowzers")
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, False)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, True)) == form
    assert json.loads(form.tojson(False, True)) == {"class": "RegularArray", "content": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}, "form_key": None}, "size": 10, "has_identities": False, "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert json.loads(str(form)) == {"class": "RegularArray", "content": "float64", "size": 10, "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert json.loads(str(form.content)) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert form.size == 10
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]
    assert form.form_key == "yowzers"

    form = awkward1.forms.UnionForm("i8", "i64", [awkward1.forms.NumpyForm([], 8, "d"), awkward1.forms.NumpyForm([], 1, "?")], parameters={"hey": ["you"]}, form_key="yowzers")
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, False)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, True)) == form
    assert json.loads(form.tojson(False, True)) == {"class": "UnionArray8_64", "tags": "i8", "index": "i64", "contents": [{"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}, "form_key": None}, {"class": "NumpyArray", "inner_shape": [], "itemsize": 1, "format": "?", "primitive": "bool", "has_identities": False, "parameters": {}, "form_key": None}], "has_identities": False, "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert json.loads(str(form)) == {"class": "UnionArray8_64", "tags": "i8", "index": "i64", "contents": ["float64", "bool"], "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert form.tags == "i8"
    assert form.index == "i64"
    assert json.loads(str(form.contents)) == [{"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}, {"class": "NumpyArray", "itemsize": 1, "format": "?", "primitive": "bool"}]
    assert json.loads(str(form.content(0))) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert json.loads(str(form.content(1))) == {"class": "NumpyArray", "itemsize": 1, "format": "?", "primitive": "bool"}
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]
    assert form.form_key == "yowzers"

    form = awkward1.forms.UnmaskedForm(awkward1.forms.NumpyForm([], 8, "d"), parameters={"hey": ["you"]}, form_key="yowzers")
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, False)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, True)) == form
    assert json.loads(form.tojson(False, True)) == {"class": "UnmaskedArray", "content": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}, "form_key": None}, "has_identities": False, "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert json.loads(str(form)) == {"class": "UnmaskedArray", "content": "float64", "parameters": {"hey": ["you"]}, "form_key": "yowzers"}
    assert json.loads(str(form.content)) == {"class": "NumpyArray", "itemsize": 8, "format": "d", "primitive": "float64"}
    assert form.has_identities == False
    assert form.parameters == {"hey": ["you"]}
    assert form.parameter("hey") == ["you"]
    assert form.form_key == "yowzers"

    form = awkward1.forms.VirtualForm(awkward1.forms.NumpyForm([], 8, "d"), True)
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, False)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, True)) == form
    assert form.form.inner_shape == []
    assert form.form.itemsize == 8
    assert form.form.primitive == "float64"
    assert form.form.has_identities == False
    assert form.form.parameters == {}
    assert form.has_length is True
    assert form.parameters == {}
    assert json.loads(form.tojson(False, True)) == {"class": "VirtualArray", "form": {"class": "NumpyArray", "inner_shape": [], "itemsize": 8, "format": "d", "primitive": "float64", "has_identities": False, "parameters": {}, "form_key": None}, "has_length": True, "has_identities": False, "parameters": {}, "form_key": None}
    assert json.loads(str(form)) == {"class": "VirtualArray", "form": "float64", "has_length": True}

    form = awkward1.forms.VirtualForm(None, False)
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, False)) == form
    assert awkward1.forms.Form.fromjson(form.tojson(False, True)) == form
    assert form.form is None
    assert form.has_length is False
    assert form.parameters == {}
    assert json.loads(form.tojson(False, True)) == {"class": "VirtualArray", "form": None, "has_length": False, "has_identities": False, "parameters": {}, "form_key": None}
    assert json.loads(str(form)) == {"class": "VirtualArray", "form": None, "has_length": False}

def fcn():
    return awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))

def test_basic():
    generator = awkward1.layout.ArrayGenerator(fcn, form=awkward1.forms.NumpyForm([], 8, "d"), length=5)

    d = awkward1._util.MappingProxy({})
    cache = awkward1.layout.ArrayCache(d)

    virtualarray = awkward1.layout.VirtualArray(generator, cache)
    assert virtualarray.peek_array is None
    assert virtualarray.array is not None
    assert awkward1.to_list(virtualarray.peek_array) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.to_list(virtualarray.array) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.to_list(d[virtualarray.cache_key]) == [1.1, 2.2, 3.3, 4.4, 5.5]

    cache = awkward1.layout.ArrayCache(None)

    virtualarray = awkward1.layout.VirtualArray(generator, cache)
    assert virtualarray.peek_array is None
    assert virtualarray.array is not None
    assert virtualarray.peek_array is None
    assert awkward1.to_list(virtualarray.array) == [1.1, 2.2, 3.3, 4.4, 5.5]

def test_slice():
    generator = awkward1.layout.ArrayGenerator(
        lambda: awkward1.Array([[1.1, 2.2, 3.3, 4.4, 5.5], [6.6, 7.7, 8.8], [100, 200, 300, 400]]),
        length=3)
    virtualarray = awkward1.layout.VirtualArray(generator)

    assert isinstance(virtualarray, awkward1.layout.VirtualArray)

    sliced = virtualarray[:-1]
    assert isinstance(sliced, awkward1.layout.VirtualArray)

    assert isinstance(sliced[1], awkward1.layout.NumpyArray)

def test_field():
    generator = awkward1.layout.ArrayGenerator(
        lambda: awkward1.Array([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}]))
    virtualarray = awkward1.layout.VirtualArray(generator)

    assert isinstance(virtualarray, awkward1.layout.VirtualArray)

    sliced = virtualarray["y"]
    assert isinstance(sliced, awkward1.layout.VirtualArray)

    assert isinstance(sliced[1], awkward1.layout.NumpyArray)

def test_single_level():
    template = awkward1.Array([[{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}], [], [{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}]])
    generator = awkward1.layout.ArrayGenerator(lambda: template, form=template.layout.form, length=3)
    d = awkward1._util.MappingProxy({})
    cache = awkward1.layout.ArrayCache(d)
    virtualarray = awkward1.layout.VirtualArray(generator, cache)

    a = virtualarray[2]
    assert isinstance(a, awkward1.layout.RecordArray)
    assert len(d) == 1
    assert awkward1.to_list(a) == [{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}]
    d.clear()

    a = virtualarray[1:]
    assert isinstance(a, awkward1.layout.VirtualArray)
    assert len(d) == 0
    b = a[1]
    assert isinstance(b, awkward1.layout.RecordArray)
    assert len(d) == 1
    assert awkward1.to_list(b) == [{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}]
    d.clear()

    a = virtualarray[[0, 2, 1, 0]]
    assert isinstance(a, awkward1.layout.VirtualArray)
    assert len(d) == 0
    b = a[1]
    assert isinstance(b, awkward1.layout.RecordArray)
    assert len(d) == 1
    assert awkward1.to_list(b) == [{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}]
    d.clear()

    a = virtualarray[[False, True, True]]
    assert isinstance(a, awkward1.layout.VirtualArray)
    assert len(d) == 0
    b = a[1]
    assert isinstance(b, awkward1.layout.RecordArray)
    assert len(d) == 1
    assert awkward1.to_list(b) == [{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}]
    d.clear()

    a = virtualarray["x"]
    assert isinstance(a, awkward1.layout.VirtualArray)
    assert len(d) == 0
    b = a[2]
    assert isinstance(b, awkward1.layout.NumpyArray)
    assert len(d) == 1
    assert awkward1.to_list(b) == [3.3, 4.4]
    d.clear()

    a = virtualarray["y"]
    assert isinstance(a, awkward1.layout.VirtualArray)
    assert len(d) == 0
    b = a[2]
    assert isinstance(b, (awkward1.layout.ListArray64, awkward1.layout.ListOffsetArray64))
    assert len(d) == 1
    assert awkward1.to_list(b) == [[3, 3, 3], [4, 4, 4, 4]]
    d.clear()

    a = virtualarray[::2, 1]
    assert isinstance(a, (awkward1.layout.RecordArray, awkward1.layout.IndexedArray64))
    assert len(d) == 1
    assert awkward1.to_list(a) == [{"x": 1.1, "y": [1]}, {"x": 4.4, "y": [4, 4, 4, 4]}]
    d.clear()

def test_iter():
    generator = awkward1.layout.ArrayGenerator(fcn, form=awkward1.forms.NumpyForm([], 8, "d"), length=5)
    d = awkward1._util.MappingProxy({})
    cache = awkward1.layout.ArrayCache(d)
    virtualarray = awkward1.layout.VirtualArray(generator, cache)

    assert len(d) == 0
    it = iter(virtualarray)
    assert len(d) == 1
    d.clear()
    assert len(d) == 0
    assert next(it) == 1.1
    assert len(d) == 0
    assert list(it) == [2.2, 3.3, 4.4, 5.5]
    assert len(d) == 0

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

    assert counter == [0, 0]

    tmp1 = virtual2[2]
    assert isinstance(tmp1, awkward1.layout.VirtualArray)
    assert counter == [1, 0]

    tmp2 = tmp1[1]
    assert tmp2 == 4.4
    assert counter == [1, 1]

def test_highlevel():
    array = awkward1.virtual(lambda: [[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    assert isinstance(array.layout, awkward1.layout.VirtualArray)
    assert awkward1.to_list(array) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    counter = [0]
    def generate():
        counter[0] += 1
        return [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    array = awkward1.virtual(generate, length=3, form={"class": "ListOffsetArray64",
                                                       "offsets": "i64",
                                                       "content": "float64"})
    assert counter[0] == 0

    assert len(array) == 3
    assert counter[0] == 0

    assert str(awkward1.type(array)) == "3 * var * float64"
    assert counter[0] == 0

    assert awkward1.to_list(array[2]) == [4.4, 5.5]
    assert counter[0] == 1
