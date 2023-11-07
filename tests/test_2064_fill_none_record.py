# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest  # noqa: F401

import awkward as ak


def test_axis_none():
    record = ak.zip({"x": [1, None], "y": [2, 3]})
    assert ak.fill_none(record, 0, axis=None).to_list() == [
        {"x": 1, "y": 2},
        {"x": 0, "y": 3},
    ]


def test_axis_last():
    record = ak.zip({"x": [1, None], "y": [2, 3]})
    assert ak.fill_none(record, 0, axis=-1).to_list() == [
        {"x": 1, "y": 2},
        {"x": 0, "y": 3},
    ]


def test_option_outside_record():
    record = ak.zip({"x": [1, 4], "y": [2, 3]}).mask[[True, False]]
    assert ak.fill_none(record, 0, axis=-1).to_list() == [{"x": 1, "y": 2}, 0]
