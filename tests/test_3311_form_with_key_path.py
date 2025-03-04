from __future__ import annotations

import numpy as np

import awkward as ak


def test_record_tuple():
    form = ak.forms.from_dict(
        {
            "class": "RecordArray",
            "fields": None,
            "contents": [
                {"class": "NumpyArray", "primitive": "int64", "form_key": "('0',)"},
                {"class": "NumpyArray", "primitive": "int64", "form_key": "('1',)"},
            ],
            "form_key": "()",
        }
    )
    array = ak.Array([(1, 2)])
    assert array.layout.form_with_key_path() == form


def test_record_dict():
    form = ak.forms.from_dict(
        {
            "class": "RecordArray",
            "fields": ["foo", "bar"],
            "contents": [
                {"class": "NumpyArray", "primitive": "int64", "form_key": "('foo',)"},
                {"class": "NumpyArray", "primitive": "int64", "form_key": "('bar',)"},
            ],
            "form_key": "()",
        }
    )
    array = ak.Array({"foo": [1], "bar": [2]})
    assert array.layout.form_with_key_path() == form


def test_numpy():
    form = ak.forms.from_dict(
        {"class": "NumpyArray", "primitive": "int64", "form_key": "()"}
    )
    array = ak.Array([1, 2, 3])
    assert array.layout.form_with_key_path() == form


def test_listoffset():
    form = ak.forms.from_dict(
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "(None,)",
            },
            "form_key": "()",
        }
    )
    array = ak.Array([[1, 2], [3]])
    assert array.layout.form_with_key_path() == form


def test_empty():
    form = ak.forms.from_dict({"class": "EmptyArray", "form_key": "()"})
    array = ak.Array([])
    assert array.layout.form_with_key_path() == form


def test_bitmasked():
    form = ak.forms.from_dict(
        {
            "class": "BitMaskedArray",
            "mask": "u8",
            "valid_when": True,
            "lsb_order": False,
            "content": {
                "class": "NumpyArray",
                "primitive": "float64",
                "form_key": "(None,)",
            },
            "form_key": "()",
        }
    )
    content = ak.Array([0.0, 1.1, 2.2, 3.3, 4.4]).layout
    mask = ak.index.IndexU8(
        np.packbits(np.array([False, True, True, False, False], dtype=np.int8))
    )
    bitmaskedarray = ak.contents.BitMaskedArray(mask, content, True, 5, False)
    assert bitmaskedarray.form_with_key_path() == form


def test_bytemasked():
    form = ak.forms.from_dict(
        {
            "class": "ByteMaskedArray",
            "mask": "i8",
            "valid_when": True,
            "content": {
                "class": "NumpyArray",
                "primitive": "float64",
                "form_key": "(None,)",
            },
            "form_key": "()",
        }
    )
    content = ak.Array([0.0, 1.1, 2.2, 3.3, 4.4]).layout
    mask = ak.index.Index8(np.array([False, True, True, False, False], dtype=np.int8))
    bytemaskedarray = ak.contents.ByteMaskedArray(mask, content, True)
    assert bytemaskedarray.form_with_key_path() == form


def test_indexedarray():
    form = ak.forms.from_dict(
        {
            "class": "IndexedArray",
            "index": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "float64",
                "form_key": "(None,)",
            },
            "form_key": "()",
        }
    )

    content = ak.Array([0.0, 1.1, 2.2, 3.3, 4.4]).layout
    index = ak.index.Index64(np.array([3, 1, 1, 4, 2], dtype=np.int64))
    indexedarray = ak.contents.IndexedArray(index, content)
    assert indexedarray.form_with_key_path() == form


def test_indexedoptionarray():
    form = ak.forms.from_dict(
        {
            "class": "IndexedOptionArray",
            "index": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "float64",
                "form_key": "(None,)",
            },
            "form_key": "()",
        }
    )
    content = ak.Array(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    ).layout
    index = ak.index.Index64(np.array([2, -1, 4, 0, 8], dtype=np.int64))
    layout = ak.Array(ak.contents.IndexedOptionArray(index, content)).layout
    assert layout.form_with_key_path() == form


def test_listarray():
    form = ak.forms.from_dict(
        {
            "class": "ListArray",
            "starts": "i64",
            "stops": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "float64",
                "form_key": "(None,)",
            },
            "form_key": "()",
        }
    )
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts = ak.index.Index64(np.array([0, 3, 3, 5, 6]))
    stops = ak.index.Index64(np.array([3, 3, 5, 6, 9]))
    layout = ak.contents.ListArray(starts, stops, content)
    assert layout.form_with_key_path() == form


def test_regulararray():
    form = ak.forms.from_dict(
        {
            "class": "RegularArray",
            "size": 3,
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "(None,)",
            },
            "form_key": "()",
        }
    )
    content = ak.Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).layout
    layout = ak.contents.RegularArray(content, 3, zeros_length=0)
    assert layout.form_with_key_path() == form


def test_unionarray():
    form = ak.forms.from_dict(
        {
            "class": "UnionArray",
            "tags": "i8",
            "index": "i64",
            "contents": [
                {
                    "class": "ListOffsetArray",
                    "offsets": "i64",
                    "content": {
                        "class": "NumpyArray",
                        "primitive": "float64",
                        "form_key": "(0, None)",
                    },
                    "form_key": "(0,)",
                },
                {
                    "class": "ListOffsetArray",
                    "offsets": "i64",
                    "content": {
                        "class": "ListOffsetArray",
                        "offsets": "i64",
                        "content": {
                            "class": "NumpyArray",
                            "primitive": "uint8",
                            "parameters": {"__array__": "char"},
                            "form_key": "(1, None, None)",
                        },
                        "parameters": {"__array__": "string"},
                        "form_key": "(1, None)",
                    },
                    "form_key": "(1,)",
                },
            ],
            "form_key": "()",
        }
    )
    content1 = ak.operations.from_iter([[], [1.1], [2.2, 2.2]], highlevel=False)
    content2 = ak.operations.from_iter([["two", "two"], ["one"], []], highlevel=False)
    tags = ak.index.Index8(np.array([0, 1, 0, 1, 0, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0, 1, 1, 2, 2], dtype=np.int64))
    layout = ak.contents.UnionArray(tags, index, [content1, content2])
    assert layout.form_with_key_path() == form


def test_unmaskedarray():
    form = ak.forms.from_dict(
        {
            "class": "UnmaskedArray",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "(None,)",
            },
            "form_key": "()",
        }
    )

    content = ak.Array([1, 2, 3, 4, 5]).layout
    layout = ak.contents.UnmaskedArray(content)
    assert layout.form_with_key_path() == form
