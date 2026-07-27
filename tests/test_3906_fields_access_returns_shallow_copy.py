# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def test_highlevel_array():
    array = ak.Array(
        {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
        }
    )

    fields = array.fields
    assert fields == ["x", "y"]
    fields.remove("x")
    assert fields == ["y"]
    assert array.fields == ["x", "y"]


def test_lowlevel_layout():
    layout = ak.contents.RecordArray(
        [ak.contents.NumpyArray([1, 2, 3]), ak.contents.NumpyArray([4, 5, 6])],
        ["x", "y"],
    )

    fields = layout.fields
    assert fields == ["x", "y"]
    fields.remove("x")
    assert fields == ["y"]
    assert layout.fields == ["y"]


def test_highlevel_record():
    record = ak.Record(
        {
            "x": 1,
            "y": 4,
        }
    )

    fields = record.fields
    assert fields == ["x", "y"]
    fields.remove("x")
    assert fields == ["y"]
    assert record.fields == ["x", "y"]


def test_lowlevel_record():
    record = ak.record.Record(
        ak.contents.RecordArray(
            [ak.contents.NumpyArray([1, 2, 3]), ak.contents.NumpyArray([4, 5, 6])],
            ["x", "y"],
        ),
        0,
    )

    fields = record.fields
    assert fields == ["x", "y"]
    fields.remove("x")
    assert fields == ["y"]
    assert record.fields == ["y"]
