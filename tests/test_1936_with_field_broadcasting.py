# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak


def test_scalar():
    array = ak.Array({"x": [1, 2, 3]})
    array["x"] = 4
    assert array.to_list() == [{"x": 4}, {"x": 4}, {"x": 4}]


def test_array():
    array = ak.Array({"x": [1, 2, 3]})
    array["x"] = [4]
    assert array.to_list() == [{"x": 4}, {"x": 4}, {"x": 4}]


def test_where_non_sequence():
    array = ak.Array({"x": [{"y": 1}, {"y": 2}, {"y": 3}]})
    result = ak.with_field(array, 4, ("x", "y"))
    assert result.to_list() == [{"x": {"y": 4}}, {"x": {"y": 4}}, {"x": {"y": 4}}]

    with pytest.raises(TypeError, match=r"New fields may only be assigned "):
        result = ak.with_field(array, 4, iter(("x", "y")))
