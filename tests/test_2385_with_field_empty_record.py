# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak


def test_no_fields():
    no_fields = ak.Array([{}, {}, {}, {}, {}])
    no_fields["new_field"] = ak.Array([1, 2, 3, 4, 5])
    assert no_fields.to_list() == [
        {"new_field": 1},
        {"new_field": 2},
        {"new_field": 3},
        {"new_field": 4},
        {"new_field": 5},
    ]


def test_union_partial_record():
    no_fields = ak.Array([{}, []])
    with pytest.raises(ValueError, match="cannot add a new field"):
        no_fields["new_field"] = ak.Array([1, 2, 3, 4, 5])


def test_union_record():
    no_fields = ak.Array([{"x": 1}, {"y": 2}, {}, {}, {}])
    no_fields["new_field"] = ak.Array([1, 2, 3, 4, 5])
    assert no_fields.to_list() == [
        {"new_field": 1, "x": 1, "y": None},
        {"new_field": 2, "x": None, "y": 2},
        {"new_field": 3, "x": None, "y": None},
        {"new_field": 4, "x": None, "y": None},
        {"new_field": 5, "x": None, "y": None},
    ]
