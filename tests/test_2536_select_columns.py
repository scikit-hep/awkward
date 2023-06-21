# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak


def test():
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

    assert form.select_columns(["*"]) == form
    assert form.select_columns(["x"]) == form
    assert form.select_columns(["x.y"]) == form
    assert form.select_columns(["x.*"]) == form
    assert form.select_columns(["x.y.*"]) == form
    assert form.select_columns(["x.y.z", "x.y.w"]) == form
    assert form.select_columns(["x.y.z"]) == ak.forms.from_dict(
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
    assert form.select_columns(["x.y.q"]) == ak.forms.from_dict(
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
    assert form.select_columns([]) == ak.forms.from_dict(
        {"class": "RecordArray", "fields": [], "contents": []}
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
