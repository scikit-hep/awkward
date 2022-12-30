# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


def test_new_style_record():
    form = {
        "class": "RecordArray",
        "fields": ["z", "y"],
        "contents": [
            {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": "node1",
            },
            {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": "node2",
            },
        ],
        "parameters": {},
        "form_key": "node0",
    }

    array = ak.from_buffers(
        form,
        1,
        {
            "node1-data": np.array([1], dtype=np.int64),
            "node2-data": np.array([2], dtype=np.int64),
        },
    )
    assert not array.is_tuple
    assert array.fields == ["z", "y"]
    assert array.to_list() == [{"z": 1, "y": 2}]


def test_new_style_tuple():
    form = {
        "class": "RecordArray",
        "fields": None,
        "contents": [
            {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": "node1",
            },
            {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": "node2",
            },
        ],
        "parameters": {},
        "form_key": "node0",
    }

    array = ak.from_buffers(
        form,
        1,
        {
            "node1-data": np.array([1], dtype=np.int64),
            "node2-data": np.array([2], dtype=np.int64),
        },
    )
    assert array.is_tuple
    assert array.fields == ["0", "1"]
    assert array.to_list() == [(1, 2)]


def test_old_style_record():
    form = {
        "class": "RecordArray",
        "contents": {
            "z": {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": "node1",
            },
            "y": {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": "node2",
            },
        },
        "parameters": {},
        "form_key": "node0",
    }

    array = ak.from_buffers(
        form,
        1,
        {
            "node1-data": np.array([1], dtype=np.int64),
            "node2-data": np.array([2], dtype=np.int64),
        },
    )
    assert not array.is_tuple
    assert array.fields == ["z", "y"]
    assert array.to_list() == [{"z": 1, "y": 2}]


def test_old_style_tuple():
    form = {
        "class": "RecordArray",
        "contents": [
            {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": "node1",
            },
            {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": "node2",
            },
        ],
        "parameters": {},
        "form_key": "node0",
    }

    array = ak.from_buffers(
        form,
        1,
        {
            "node1-data": np.array([1], dtype=np.int64),
            "node2-data": np.array([2], dtype=np.int64),
        },
    )
    assert array.is_tuple
    assert array.fields == ["0", "1"]
    assert array.to_list() == [(1, 2)]


def test_old_style_contents_new_style_list():
    form = {
        "class": "RecordArray",
        "fields": ["x", "y"],
        "contents": {
            "x": {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": "node1",
            },
            "y": {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": "node2",
            },
        },
        "parameters": {},
        "form_key": "node0",
    }

    with pytest.raises(
        TypeError, match=".*new-style RecordForm contents must not be mappings.*"
    ):
        ak.from_buffers(
            form,
            1,
            {
                "node1-data": np.array([1], dtype=np.int64),
                "node2-data": np.array([2], dtype=np.int64),
            },
        )
