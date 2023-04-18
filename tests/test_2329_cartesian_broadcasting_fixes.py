# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_nested_exis_0():
    arrays = {"x": np.arange(4), "y": ["this", "that", "foo", "bar!"]}

    result = ak.cartesian(arrays, nested=True, axis=0)
    assert result.to_list() == [
        [
            {"x": 0, "y": "this"},
            {"x": 0, "y": "that"},
            {"x": 0, "y": "foo"},
            {"x": 0, "y": "bar!"},
        ],
        [
            {"x": 1, "y": "this"},
            {"x": 1, "y": "that"},
            {"x": 1, "y": "foo"},
            {"x": 1, "y": "bar!"},
        ],
        [
            {"x": 2, "y": "this"},
            {"x": 2, "y": "that"},
            {"x": 2, "y": "foo"},
            {"x": 2, "y": "bar!"},
        ],
        [
            {"x": 3, "y": "this"},
            {"x": 3, "y": "that"},
            {"x": 3, "y": "foo"},
            {"x": 3, "y": "bar!"},
        ],
    ]

    result = ak.cartesian(arrays, nested=["x"], axis=0)
    assert result.to_list() == [
        [
            {"x": 0, "y": "this"},
            {"x": 0, "y": "that"},
            {"x": 0, "y": "foo"},
            {"x": 0, "y": "bar!"},
        ],
        [
            {"x": 1, "y": "this"},
            {"x": 1, "y": "that"},
            {"x": 1, "y": "foo"},
            {"x": 1, "y": "bar!"},
        ],
        [
            {"x": 2, "y": "this"},
            {"x": 2, "y": "that"},
            {"x": 2, "y": "foo"},
            {"x": 2, "y": "bar!"},
        ],
        [
            {"x": 3, "y": "this"},
            {"x": 3, "y": "that"},
            {"x": 3, "y": "foo"},
            {"x": 3, "y": "bar!"},
        ],
    ]
