# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json

try:
    # pybind11 only supports cPickle protocol 2+ (-1 in pickle.dumps)
    # (automatically satisfied in Python 3; this is just to keep testing Python 2.7)
    import cPickle as pickle
except ImportError:
    import pickle

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2  # noqa: F401

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_forms():
    form = ak._v2.forms.NumpyForm("float64")
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert form.inner_shape == ()
    assert form.itemsize == 8
    assert form.primitive == "float64"
    assert form.has_identifier is False
    assert form.parameters == {}
    assert form.form_key is None
    assert json.loads(form.to_json()) == {
        "class": "NumpyArray",
        "inner_shape": [],
        "primitive": "float64",
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert json.loads(str(form)) == {
        "class": "NumpyArray",
        "primitive": "float64",
    }

    form = ak._v2.forms.NumpyForm(
        "float64",
        [1, 2, 3],
        has_identifier=True,
        parameters={"hey": ["you", {"there": 3}]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert form.inner_shape == (1, 2, 3)
    assert form.itemsize == 8
    assert form.primitive == "float64"
    assert form.has_identifier is True
    assert form.parameters == {"hey": ["you", {"there": 3}]}
    assert form.form_key == "yowzers"
    assert json.loads(form.to_json()) == {
        "class": "NumpyArray",
        "inner_shape": [1, 2, 3],
        "primitive": "float64",
        "has_identifier": True,
        "parameters": {"hey": ["you", {"there": 3}]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "NumpyArray",
        "inner_shape": [1, 2, 3],
        "primitive": "float64",
        "has_identifier": True,
        "parameters": {"hey": ["you", {"there": 3}]},
        "form_key": "yowzers",
    }

    form = ak._v2.forms.BitMaskedForm(
        "i8",
        ak._v2.forms.NumpyForm("float64"),
        True,
        False,
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "BitMaskedArray",
        "mask": "i8",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "valid_when": True,
        "lsb_order": False,
        "has_identifier": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "BitMaskedArray",
        "mask": "i8",
        "content": "float64",
        "valid_when": True,
        "lsb_order": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak._v2.forms.EmptyForm(
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "EmptyArray",
        "has_identifier": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "EmptyArray",
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak._v2.forms.IndexedForm(
        "i64",
        ak._v2.forms.NumpyForm("float64"),
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "IndexedArray",
        "index": "i64",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "IndexedArray",
        "index": "i64",
        "content": "float64",
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak._v2.forms.IndexedOptionForm(
        "i64",
        ak._v2.forms.NumpyForm("float64"),
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "IndexedOptionArray",
        "index": "i64",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "IndexedOptionArray",
        "index": "i64",
        "content": "float64",
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak._v2.forms.ListForm(
        "i64",
        "i64",
        ak._v2.forms.NumpyForm("float64"),
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "ListArray",
        "starts": "i64",
        "stops": "i64",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "ListArray",
        "starts": "i64",
        "stops": "i64",
        "content": "float64",
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak._v2.forms.ListOffsetForm(
        "i64",
        ak._v2.forms.NumpyForm("float64"),
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "ListOffsetArray",
        "offsets": "i64",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "ListOffsetArray",
        "offsets": "i64",
        "content": "float64",
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak._v2.forms.RecordForm(
        [ak._v2.forms.NumpyForm("float64"), ak._v2.forms.NumpyForm("bool")],
        ["one", "two"],
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "RecordArray",
        "contents": {
            "one": {
                "class": "NumpyArray",
                "inner_shape": [],
                "primitive": "float64",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
            "two": {
                "class": "NumpyArray",
                "inner_shape": [],
                "primitive": "bool",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
        },
        "has_identifier": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "RecordArray",
        "contents": {
            "one": "float64",
            "two": "bool",
        },
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak._v2.forms.RecordForm(
        [ak._v2.forms.NumpyForm("float64"), ak._v2.forms.NumpyForm("bool")],
        None,
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "RecordArray",
        "contents": [
            {
                "class": "NumpyArray",
                "inner_shape": [],
                "primitive": "float64",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "inner_shape": [],
                "primitive": "bool",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
        ],
        "has_identifier": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "RecordArray",
        "contents": ["float64", "bool"],
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak._v2.forms.RegularForm(
        ak._v2.forms.NumpyForm("float64"),
        10,
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "RegularArray",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "size": 10,
        "has_identifier": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "RegularArray",
        "content": "float64",
        "size": 10,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak._v2.forms.UnionForm(
        "i8",
        "i64",
        [ak._v2.forms.NumpyForm("float64"), ak._v2.forms.NumpyForm("bool")],
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i64",
        "contents": [
            {
                "class": "NumpyArray",
                "inner_shape": [],
                "primitive": "float64",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "inner_shape": [],
                "primitive": "bool",
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
        ],
        "has_identifier": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i64",
        "contents": ["float64", "bool"],
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak._v2.forms.UnmaskedForm(
        ak._v2.forms.NumpyForm("float64"),
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "UnmaskedArray",
        "content": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_identifier": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "UnmaskedArray",
        "content": "float64",
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }

    form = ak._v2.forms.VirtualForm(
        ak._v2.forms.NumpyForm("float64"),
        True,
        parameters={"hey": ["you"]},
        form_key="yowzers",
    )
    assert form == form
    assert pickle.loads(pickle.dumps(form, -1)) == form
    assert ak._v2.forms.from_json(form.to_json()) == form
    assert json.loads(form.to_json()) == {
        "class": "VirtualArray",
        "form": {
            "class": "NumpyArray",
            "inner_shape": [],
            "primitive": "float64",
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        },
        "has_length": True,
        "has_identifier": False,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }
    assert json.loads(str(form)) == {
        "class": "VirtualArray",
        "form": "float64",
        "has_length": True,
        "parameters": {"hey": ["you"]},
        "form_key": "yowzers",
    }


def fcn():
    return ak._v2.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))


def test_basic():
    generator = ak._v2.contents.FunctionGenerator(
        5, ak._v2.forms.NumpyForm("float64"), fcn
    )
    cache = {}

    virtualarray = ak._v2.contents.VirtualArray(generator, cache)
    assert virtualarray.peek_array is None
    assert virtualarray.array is not None
    assert ak.to_list(virtualarray.peek_array) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert ak.to_list(virtualarray.array) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert ak.to_list(cache[virtualarray.cache_key]) == [1.1, 2.2, 3.3, 4.4, 5.5]

    cache = None

    virtualarray = ak._v2.contents.VirtualArray(generator, cache)
    assert virtualarray.peek_array is None
    assert virtualarray.array is not None
    assert virtualarray.peek_array is None
    assert ak.to_list(virtualarray.array) == [1.1, 2.2, 3.3, 4.4, 5.5]


def test_slice():
    generator = ak._v2.contents.FunctionGenerator(
        3,
        None,
        lambda: v1_to_v2(
            ak.Array(
                [[1.1, 2.2, 3.3, 4.4, 5.5], [6.6, 7.7, 8.8], [100, 200, 300, 400]]
            ).layout
        ),
    )
    virtualarray = ak._v2.contents.VirtualArray(generator)

    assert isinstance(virtualarray, ak._v2.contents.VirtualArray)

    sliced = virtualarray[:-1]

    assert isinstance(sliced, ak._v2.contents.VirtualArray)

    assert isinstance(sliced[1], ak._v2.contents.NumpyArray)


def test_field():
    generator = ak._v2.contents.FunctionGenerator(
        None,
        None,
        lambda: v1_to_v2(
            ak.Array(
                [
                    {"x": 0.0, "y": []},
                    {"x": 1.1, "y": [1]},
                    {"x": 2.2, "y": [2, 2]},
                    {"x": 3.3, "y": [3, 3, 3]},
                ]
            ).layout
        ),
    )
    virtualarray = ak._v2.contents.VirtualArray(generator)

    assert isinstance(virtualarray, ak._v2.contents.VirtualArray)

    sliced = virtualarray["y"]
    assert isinstance(sliced, ak._v2.contents.VirtualArray)

    assert isinstance(sliced[1], ak._v2.contents.NumpyArray)


def test_fields():
    generator = ak._v2.contents.FunctionGenerator(
        None,
        None,
        lambda: v1_to_v2(
            ak.Array(
                [
                    {"x": 0.0, "y": []},
                    {"x": 1.1, "y": [1]},
                    {"x": 2.2, "y": [2, 2]},
                    {"x": 3.3, "y": [3, 3, 3]},
                ]
            ).layout
        ),
    )
    virtualarray = ak._v2.contents.VirtualArray(generator)

    assert isinstance(virtualarray, ak._v2.contents.VirtualArray)

    sliced = virtualarray[["y", "x"]]
    assert isinstance(sliced, ak._v2.contents.VirtualArray)

    assert isinstance(sliced[1], ak._v2.record.Record)


def test_single_level():
    template = v1_to_v2(
        ak.Array(
            [
                [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}],
                [],
                [{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}],
            ]
        ).layout
    )
    generator = ak._v2.contents.FunctionGenerator(3, template.form, lambda: template)
    cache = {}
    virtualarray = ak._v2.contents.VirtualArray(generator, cache)

    a = virtualarray[2]
    assert isinstance(a, ak._v2.contents.RecordArray)
    assert len(cache) == 1
    assert ak.to_list(a) == [{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}]
    cache.clear()

    a = virtualarray[1:]
    assert isinstance(a, ak._v2.contents.VirtualArray)
    assert len(cache) == 0
    b = a[1]
    assert isinstance(b, ak._v2.contents.RecordArray)
    assert len(cache) >= 1
    assert ak.to_list(b) == [{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}]
    cache.clear()

    a = virtualarray[[0, 2, 1, 0]]
    assert isinstance(a, ak._v2.contents.VirtualArray)
    assert len(cache) == 0
    b = a[1]
    assert isinstance(b, ak._v2.contents.RecordArray)
    assert len(cache) >= 1
    assert ak.to_list(b) == [{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}]
    cache.clear()

    a = virtualarray[[False, True, True]]
    assert isinstance(a, ak._v2.contents.VirtualArray)
    assert len(cache) == 0
    b = a[1]
    assert isinstance(b, ak._v2.contents.RecordArray)
    assert len(cache) >= 1
    assert ak.to_list(b) == [{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}]
    cache.clear()

    a = virtualarray["x"]
    assert isinstance(a, ak._v2.contents.VirtualArray)
    assert len(cache) == 0
    b = a[2]
    assert isinstance(b, ak._v2.contents.NumpyArray)
    assert len(cache) >= 1
    assert ak.to_list(b) == [3.3, 4.4]
    cache.clear()

    a = virtualarray["y"]
    assert isinstance(a, ak._v2.contents.VirtualArray)
    assert len(cache) == 0
    b = a[2]
    assert isinstance(b, (ak._v2.contents.ListArray, ak._v2.contents.ListOffsetArray))
    assert len(cache) >= 1
    assert ak.to_list(b) == [[3, 3, 3], [4, 4, 4, 4]]
    cache.clear()

    a = virtualarray[:1, 1]
    assert isinstance(a, (ak._v2.contents.RecordArray, ak._v2.contents.IndexedArray))
    assert len(cache) >= 1
    assert ak.to_list(a) == [{"x": 1.1, "y": [1]}]
    cache.clear()

    a = virtualarray[::2, 1]
    assert isinstance(a, (ak._v2.contents.RecordArray, ak._v2.contents.IndexedArray))
    assert len(cache) >= 1
    assert ak.to_list(a) == [{"x": 1.1, "y": [1]}, {"x": 4.4, "y": [4, 4, 4, 4]}]
    cache.clear()


def test_iter():
    generator = ak._v2.contents.FunctionGenerator(
        5, ak._v2.forms.NumpyForm("float64"), fcn
    )
    cache = {}
    virtualarray = ak._v2.contents.VirtualArray(generator, cache)

    assert len(cache) == 0
    it = iter(virtualarray)
    assert len(cache) == 0
    assert next(it) == 1.1
    assert len(cache) == 1
    assert list(it) == [2.2, 3.3, 4.4, 5.5]
    assert len(cache) == 1


def test_nested_virtualness():
    counter = [0, 0]

    content = ak._v2.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )

    def materialize1():
        counter[1] += 1
        return content

    generator1 = ak._v2.contents.FunctionGenerator(
        len(content), content.form, materialize1
    )
    virtual1 = ak._v2.contents.VirtualArray(generator1)

    offsets = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 10], dtype=np.int64))
    listarray = ak._v2.contents.ListOffsetArray(offsets, virtual1)

    def materialize2():
        counter[0] += 1
        return listarray

    generator2 = ak._v2.contents.FunctionGenerator(
        len(listarray), listarray.form, materialize2
    )
    virtual2 = ak._v2.contents.VirtualArray(generator2)

    assert counter == [0, 0]

    tmp1 = virtual2[2]
    assert isinstance(tmp1, ak._v2.contents.VirtualArray)
    assert counter == [1, 0]

    tmp2 = tmp1[1]
    assert tmp2 == 4.4
    assert counter == [1, 1]
