# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest  # noqa: F401

import awkward as ak


def test_no_prune():
    form = ak.Array(
        [
            {
                "x": [
                    {
                        "y": {
                            "z": [1, 2, 3],
                            "w": 4,
                        }
                    }
                ]
            }
        ]
    ).layout.form

    assert form.select_columns(["*"], prune_unions_and_records=False) == form
    assert form.select_columns(["x"], prune_unions_and_records=False) == form
    assert form.select_columns(["x.y"], prune_unions_and_records=False) == form
    assert form.select_columns(["x.*"], prune_unions_and_records=False) == form
    assert (
        form.select_columns(["x.y.z", "x.y.w"], prune_unions_and_records=False) == form
    )
    assert (
        form.select_columns(["x.y.z"])
        == form.select_columns(["x.y.z*"])
        == ak.forms.from_dict(
            {
                "class": "RecordArray",
                "fields": ["x"],
                "contents": [
                    {
                        "class": "ListOffsetArray",
                        "offsets": "i64",
                        "content": {
                            "class": "RecordArray",
                            "fields": ["y"],
                            "contents": [
                                {
                                    "class": "RecordArray",
                                    "fields": [
                                        "z",
                                    ],
                                    "contents": [
                                        {
                                            "class": "ListOffsetArray",
                                            "offsets": "i64",
                                            "content": "int64",
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                ],
            }
        )
    )
    assert form.select_columns(
        ["x.y.q"], prune_unions_and_records=False
    ) == ak.forms.from_dict(
        {
            "class": "RecordArray",
            "fields": ["x"],
            "contents": [
                {
                    "class": "ListOffsetArray",
                    "offsets": "i64",
                    "content": {
                        "class": "RecordArray",
                        "fields": ["y"],
                        "contents": [
                            {
                                "class": "RecordArray",
                                "fields": [],
                                "contents": [],
                            }
                        ],
                    },
                }
            ],
        }
    )
    assert form.select_columns(
        [], prune_unions_and_records=False
    ) == ak.forms.from_dict({"class": "RecordArray", "fields": [], "contents": []})

    union_form = ak.forms.from_dict(
        {
            "class": "UnionArray",
            "tags": "i8",
            "index": "i64",
            "contents": [
                {"class": "RecordArray", "fields": ["x"], "contents": ["int64"]},
                {"class": "RecordArray", "fields": ["y"], "contents": ["int64"]},
            ],
        }
    )
    assert union_form.select_columns(
        "z", prune_unions_and_records=False
    ) == ak.forms.from_dict(
        {
            "class": "UnionArray",
            "tags": "i8",
            "index": "i64",
            "contents": [
                {"class": "RecordArray", "fields": [], "contents": []},
                {"class": "RecordArray", "fields": [], "contents": []},
            ],
        }
    )


def test_prune():
    form = ak.Array(
        [
            {
                "x": [
                    {
                        "y": {
                            "z": [1, 2, 3],
                            "w": 4,
                        }
                    }
                ]
            }
        ]
    ).layout.form

    assert form.select_columns(
        ["x.y.q"], prune_unions_and_records=True
    ) == ak.forms.from_dict(
        {
            "class": "RecordArray",
            "fields": [],
            "contents": [],
        }
    )

    union_form = ak.forms.from_dict(
        {
            "class": "UnionArray",
            "tags": "i8",
            "index": "i64",
            "contents": [
                {"class": "RecordArray", "fields": ["x"], "contents": ["int64"]},
                {"class": "RecordArray", "fields": ["y"], "contents": ["int64"]},
            ],
        }
    )
    assert (
        union_form.select_columns("z", prune_unions_and_records=True)
        == ak.forms.EmptyForm()
    )


def test_very_large_record():
    form = ak.forms.from_dict(
        {
            "class": "RecordArray",
            "fields": [f"x_{i}" for i in range(10_000)],
            "contents": ["int64"] * 10_000,
        }
    )
    assert form.select_columns(["*"]) == form
    assert form.select_columns(["x*"]) == form
    assert form.select_columns(["x_[0-9]"]) == ak.forms.from_dict(
        {
            "class": "RecordArray",
            "fields": [
                "x_0",
                "x_1",
                "x_2",
                "x_3",
                "x_4",
                "x_5",
                "x_6",
                "x_7",
                "x_8",
                "x_9",
            ],
            "contents": [
                "int64",
                "int64",
                "int64",
                "int64",
                "int64",
                "int64",
                "int64",
                "int64",
                "int64",
                "int64",
            ],
        }
    )
