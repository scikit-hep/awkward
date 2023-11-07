# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_union_simplification():
    array = ak.Array(
        ak.contents.UnionArray(
            ak.index.Index8(np.arange(64, dtype=np.int8) % 2),
            ak.index.Index64(np.arange(64, dtype=np.int64) // 2),
            [
                ak.contents.RecordArray(
                    [ak.contents.NumpyArray(np.arange(64, dtype=np.int64))], ["x"]
                ),
                ak.contents.RecordArray(
                    [
                        ak.contents.NumpyArray(np.arange(64, dtype=np.int64)),
                        ak.contents.NumpyArray(np.arange(64, dtype=np.int8)),
                    ],
                    ["x", "y"],
                ),
            ],
        )
    )

    form, length, container = ak.to_buffers(array)

    assert form.to_dict() == {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i64",
        "contents": [
            {
                "class": "RecordArray",
                "fields": ["x"],
                "contents": [
                    {
                        "class": "NumpyArray",
                        "primitive": "int64",
                        "inner_shape": [],
                        "parameters": {},
                        "form_key": "node2",
                    }
                ],
                "parameters": {},
                "form_key": "node1",
            },
            {
                "class": "RecordArray",
                "fields": ["x", "y"],
                "contents": [
                    {
                        "class": "NumpyArray",
                        "primitive": "int64",
                        "inner_shape": [],
                        "parameters": {},
                        "form_key": "node4",
                    },
                    {
                        "class": "NumpyArray",
                        "primitive": "int8",
                        "inner_shape": [],
                        "parameters": {},
                        "form_key": "node5",
                    },
                ],
                "parameters": {},
                "form_key": "node3",
            },
        ],
        "parameters": {},
        "form_key": "node0",
    }

    projected_form = {
        "class": "UnionArray",
        "tags": "i8",
        "index": "i64",
        "contents": [
            {
                "class": "RecordArray",
                "fields": ["x"],
                "contents": [
                    {
                        "class": "NumpyArray",
                        "primitive": "int64",
                        "inner_shape": [],
                        "parameters": {},
                        "form_key": "node2",
                    }
                ],
                "parameters": {},
                "form_key": "node1",
            },
            {
                "class": "RecordArray",
                "fields": ["x"],
                "contents": [
                    {
                        "class": "NumpyArray",
                        "primitive": "int64",
                        "inner_shape": [],
                        "parameters": {},
                        "form_key": "node4",
                    }
                ],
                "parameters": {},
                "form_key": "node3",
            },
        ],
        "parameters": {},
        "form_key": "node0",
    }
    container.pop("node5-data")
    projected = ak.from_buffers(
        projected_form, length, container, allow_noncanonical_form=True
    )
    assert projected.layout.form.to_dict(verbose=False) == {
        "class": "IndexedArray",
        "index": "i64",
        "content": {"class": "RecordArray", "fields": ["x"], "contents": ["int64"]},
    }
    assert ak.almost_equal(array[["x"]], projected)
