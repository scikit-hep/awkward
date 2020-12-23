# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_flat():
    array = ak.Array(
        [
            {"eta": {"up": 1, "down": 2}, "phi": {"up": 3, "down": 4}, "blah": 999},
            {"eta": {"up": 5, "down": 6}, "phi": {"up": 7, "down": 8}, "blah": 999},
            {"eta": {"up": 9, "down": 10}, "phi": {"up": 11, "down": 12}, "blah": 999},
        ]
    )
    assert array[["eta", "phi"]].tolist() == [
        {"eta": {"up": 1, "down": 2}, "phi": {"up": 3, "down": 4}},
        {"eta": {"up": 5, "down": 6}, "phi": {"up": 7, "down": 8}},
        {"eta": {"up": 9, "down": 10}, "phi": {"up": 11, "down": 12}},
    ]
    assert array[["eta", "phi"], ["up", "down"]].tolist() == [
        {"eta": {"up": 1, "down": 2}, "phi": {"up": 3, "down": 4}},
        {"eta": {"up": 5, "down": 6}, "phi": {"up": 7, "down": 8}},
        {"eta": {"up": 9, "down": 10}, "phi": {"up": 11, "down": 12}},
    ]
    assert array[["eta", "phi"], "up"].tolist() == [
        {"eta": 1, "phi": 3},
        {"eta": 5, "phi": 7},
        {"eta": 9, "phi": 11},
    ]
    assert array[:, ["eta", "phi"], "down"].tolist() == [
        {"eta": 2, "phi": 4},
        {"eta": 6, "phi": 8},
        {"eta": 10, "phi": 12},
    ]
    assert array[["eta", "phi"], :, "down"].tolist() == [
        {"eta": 2, "phi": 4},
        {"eta": 6, "phi": 8},
        {"eta": 10, "phi": 12},
    ]
    assert array[["eta", "phi"], "down", :].tolist() == [
        {"eta": 2, "phi": 4},
        {"eta": 6, "phi": 8},
        {"eta": 10, "phi": 12},
    ]
    assert array[1:, ["eta", "phi"], "down"].tolist() == [
        {"eta": 6, "phi": 8},
        {"eta": 10, "phi": 12},
    ]
    assert array[["eta", "phi"], 1:, "down"].tolist() == [
        {"eta": 6, "phi": 8},
        {"eta": 10, "phi": 12},
    ]
    assert array[["eta", "phi"], "down", 1:].tolist() == [
        {"eta": 6, "phi": 8},
        {"eta": 10, "phi": 12},
    ]


def test_flat_virtual():
    array = ak.Array(
        [
            {"eta": {"up": 1, "down": 2}, "phi": {"up": 3, "down": 4}, "blah": 999},
            {"eta": {"up": 5, "down": 6}, "phi": {"up": 7, "down": 8}, "blah": 999},
            {"eta": {"up": 9, "down": 10}, "phi": {"up": 11, "down": 12}, "blah": 999},
        ]
    )
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"]
    ].tolist() == [
        {"eta": {"up": 1, "down": 2}, "phi": {"up": 3, "down": 4}},
        {"eta": {"up": 5, "down": 6}, "phi": {"up": 7, "down": 8}},
        {"eta": {"up": 9, "down": 10}, "phi": {"up": 11, "down": 12}},
    ]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"], ["up", "down"]
    ].tolist() == [
        {"eta": {"up": 1, "down": 2}, "phi": {"up": 3, "down": 4}},
        {"eta": {"up": 5, "down": 6}, "phi": {"up": 7, "down": 8}},
        {"eta": {"up": 9, "down": 10}, "phi": {"up": 11, "down": 12}},
    ]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"], "up"
    ].tolist() == [{"eta": 1, "phi": 3}, {"eta": 5, "phi": 7}, {"eta": 9, "phi": 11}]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        :, ["eta", "phi"], "down"
    ].tolist() == [{"eta": 2, "phi": 4}, {"eta": 6, "phi": 8}, {"eta": 10, "phi": 12}]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"], :, "down"
    ].tolist() == [{"eta": 2, "phi": 4}, {"eta": 6, "phi": 8}, {"eta": 10, "phi": 12}]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"], "down", :
    ].tolist() == [{"eta": 2, "phi": 4}, {"eta": 6, "phi": 8}, {"eta": 10, "phi": 12}]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        1:, ["eta", "phi"], "down"
    ].tolist() == [{"eta": 6, "phi": 8}, {"eta": 10, "phi": 12}]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"], 1:, "down"
    ].tolist() == [{"eta": 6, "phi": 8}, {"eta": 10, "phi": 12}]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"], "down", 1:
    ].tolist() == [{"eta": 6, "phi": 8}, {"eta": 10, "phi": 12}]


def test_nested():
    array = ak.Array(
        [
            [
                {"eta": {"up": 1, "down": 2}, "phi": {"up": 3, "down": 4}, "blah": 999},
                {"eta": {"up": 5, "down": 6}, "phi": {"up": 7, "down": 8}, "blah": 999},
            ],
            [],
            [
                {
                    "eta": {"up": 9, "down": 10},
                    "phi": {"up": 11, "down": 12},
                    "blah": 999,
                }
            ],
        ]
    )
    assert array[["eta", "phi"]].tolist() == [
        [
            {"eta": {"up": 1, "down": 2}, "phi": {"up": 3, "down": 4}},
            {"eta": {"up": 5, "down": 6}, "phi": {"up": 7, "down": 8}},
        ],
        [],
        [{"eta": {"up": 9, "down": 10}, "phi": {"up": 11, "down": 12}}],
    ]
    assert array[["eta", "phi"], ["up", "down"]].tolist() == [
        [
            {"eta": {"up": 1, "down": 2}, "phi": {"up": 3, "down": 4}},
            {"eta": {"up": 5, "down": 6}, "phi": {"up": 7, "down": 8}},
        ],
        [],
        [{"eta": {"up": 9, "down": 10}, "phi": {"up": 11, "down": 12}}],
    ]
    assert array[["eta", "phi"], "up"].tolist() == [
        [{"eta": 1, "phi": 3}, {"eta": 5, "phi": 7}],
        [],
        [{"eta": 9, "phi": 11}],
    ]
    assert array[:, :, ["eta", "phi"], "down"].tolist() == [
        [{"eta": 2, "phi": 4}, {"eta": 6, "phi": 8}],
        [],
        [{"eta": 10, "phi": 12}],
    ]
    assert array[:, ["eta", "phi"], :, "down"].tolist() == [
        [{"eta": 2, "phi": 4}, {"eta": 6, "phi": 8}],
        [],
        [{"eta": 10, "phi": 12}],
    ]
    assert array[:, ["eta", "phi"], "down", :].tolist() == [
        [{"eta": 2, "phi": 4}, {"eta": 6, "phi": 8}],
        [],
        [{"eta": 10, "phi": 12}],
    ]
    assert array[["eta", "phi"], :, :, "down"].tolist() == [
        [{"eta": 2, "phi": 4}, {"eta": 6, "phi": 8}],
        [],
        [{"eta": 10, "phi": 12}],
    ]
    assert array[["eta", "phi"], :, "down", :].tolist() == [
        [{"eta": 2, "phi": 4}, {"eta": 6, "phi": 8}],
        [],
        [{"eta": 10, "phi": 12}],
    ]
    assert array[["eta", "phi"], "down", :, :].tolist() == [
        [{"eta": 2, "phi": 4}, {"eta": 6, "phi": 8}],
        [],
        [{"eta": 10, "phi": 12}],
    ]
    assert array[1:, :, ["eta", "phi"], "down"].tolist() == [
        [],
        [{"eta": 10, "phi": 12}],
    ]
    assert array[1:, ["eta", "phi"], :, "down"].tolist() == [
        [],
        [{"eta": 10, "phi": 12}],
    ]
    assert array[1:, ["eta", "phi"], "down", :].tolist() == [
        [],
        [{"eta": 10, "phi": 12}],
    ]
    assert array[["eta", "phi"], 1:, :, "down"].tolist() == [
        [],
        [{"eta": 10, "phi": 12}],
    ]
    assert array[["eta", "phi"], 1:, "down", :].tolist() == [
        [],
        [{"eta": 10, "phi": 12}],
    ]
    assert array[["eta", "phi"], "down", 1:, :].tolist() == [
        [],
        [{"eta": 10, "phi": 12}],
    ]
    assert array[:, 1:, ["eta", "phi"], "down"].tolist() == [
        [{"eta": 6, "phi": 8}],
        [],
        [],
    ]
    assert array[:, ["eta", "phi"], 1:, "down"].tolist() == [
        [{"eta": 6, "phi": 8}],
        [],
        [],
    ]
    assert array[:, ["eta", "phi"], "down", 1:].tolist() == [
        [{"eta": 6, "phi": 8}],
        [],
        [],
    ]
    assert array[["eta", "phi"], :, 1:, "down"].tolist() == [
        [{"eta": 6, "phi": 8}],
        [],
        [],
    ]
    assert array[["eta", "phi"], :, "down", 1:].tolist() == [
        [{"eta": 6, "phi": 8}],
        [],
        [],
    ]
    assert array[["eta", "phi"], "down", :, 1:].tolist() == [
        [{"eta": 6, "phi": 8}],
        [],
        [],
    ]


def test_nested_virtual():
    array = ak.Array(
        [
            [
                {"eta": {"up": 1, "down": 2}, "phi": {"up": 3, "down": 4}, "blah": 999},
                {"eta": {"up": 5, "down": 6}, "phi": {"up": 7, "down": 8}, "blah": 999},
            ],
            [],
            [
                {
                    "eta": {"up": 9, "down": 10},
                    "phi": {"up": 11, "down": 12},
                    "blah": 999,
                }
            ],
        ]
    )
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"]
    ].tolist() == [
        [
            {"eta": {"up": 1, "down": 2}, "phi": {"up": 3, "down": 4}},
            {"eta": {"up": 5, "down": 6}, "phi": {"up": 7, "down": 8}},
        ],
        [],
        [{"eta": {"up": 9, "down": 10}, "phi": {"up": 11, "down": 12}}],
    ]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"], ["up", "down"]
    ].tolist() == [
        [
            {"eta": {"up": 1, "down": 2}, "phi": {"up": 3, "down": 4}},
            {"eta": {"up": 5, "down": 6}, "phi": {"up": 7, "down": 8}},
        ],
        [],
        [{"eta": {"up": 9, "down": 10}, "phi": {"up": 11, "down": 12}}],
    ]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"], "up"
    ].tolist() == [
        [{"eta": 1, "phi": 3}, {"eta": 5, "phi": 7}],
        [],
        [{"eta": 9, "phi": 11}],
    ]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        :, :, ["eta", "phi"], "down"
    ].tolist() == [
        [{"eta": 2, "phi": 4}, {"eta": 6, "phi": 8}],
        [],
        [{"eta": 10, "phi": 12}],
    ]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        :, ["eta", "phi"], :, "down"
    ].tolist() == [
        [{"eta": 2, "phi": 4}, {"eta": 6, "phi": 8}],
        [],
        [{"eta": 10, "phi": 12}],
    ]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        :, ["eta", "phi"], "down", :
    ].tolist() == [
        [{"eta": 2, "phi": 4}, {"eta": 6, "phi": 8}],
        [],
        [{"eta": 10, "phi": 12}],
    ]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"], :, :, "down"
    ].tolist() == [
        [{"eta": 2, "phi": 4}, {"eta": 6, "phi": 8}],
        [],
        [{"eta": 10, "phi": 12}],
    ]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"], :, "down", :
    ].tolist() == [
        [{"eta": 2, "phi": 4}, {"eta": 6, "phi": 8}],
        [],
        [{"eta": 10, "phi": 12}],
    ]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"], "down", :, :
    ].tolist() == [
        [{"eta": 2, "phi": 4}, {"eta": 6, "phi": 8}],
        [],
        [{"eta": 10, "phi": 12}],
    ]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        1:, :, ["eta", "phi"], "down"
    ].tolist() == [[], [{"eta": 10, "phi": 12}],]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        1:, ["eta", "phi"], :, "down"
    ].tolist() == [[], [{"eta": 10, "phi": 12}],]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        1:, ["eta", "phi"], "down", :
    ].tolist() == [[], [{"eta": 10, "phi": 12}],]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"], 1:, :, "down"
    ].tolist() == [[], [{"eta": 10, "phi": 12}],]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"], 1:, "down", :
    ].tolist() == [[], [{"eta": 10, "phi": 12}],]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"], "down", 1:, :
    ].tolist() == [[], [{"eta": 10, "phi": 12}],]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        :, 1:, ["eta", "phi"], "down"
    ].tolist() == [[{"eta": 6, "phi": 8}], [], [],]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        :, ["eta", "phi"], 1:, "down"
    ].tolist() == [[{"eta": 6, "phi": 8}], [], [],]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        :, ["eta", "phi"], "down", 1:
    ].tolist() == [[{"eta": 6, "phi": 8}], [], [],]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"], :, 1:, "down"
    ].tolist() == [[{"eta": 6, "phi": 8}], [], [],]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"], :, "down", 1:
    ].tolist() == [[{"eta": 6, "phi": 8}], [], [],]
    assert ak.virtual(lambda: array, length=3, form=array.layout.form)[
        ["eta", "phi"], "down", :, 1:
    ].tolist() == [[{"eta": 6, "phi": 8}], [], [],]
