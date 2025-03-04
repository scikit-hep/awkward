# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import os

import pytest

import awkward as ak


def array_with_dotted_fields():
    return ak.Array(
        [
            {
                "x": [
                    {
                        "y": {
                            "z": [1, 2, 3],
                            "w.1": 4,
                        }
                    }
                ]
            }
        ]
    )


def test_alternative_specifiers():
    array = array_with_dotted_fields()
    form = array.layout.form
    assert form.select_columns("*") == form
    assert form.select_columns([("x", "y", "w.1")]) == form.select_columns("x.y.w*")
    assert form.select_columns([["x", "y", "w.1"], "x.y.z"]) == form


def test_columns_with_dots_from_parquet(tmp_path):
    # ruff: noqa: F841
    _pq = pytest.importorskip("pyarrow.parquet")
    array = array_with_dotted_fields()
    parquet_file = os.path.join(tmp_path, "test_3088_array1.parquet")
    ak.to_parquet(array, parquet_file)
    array_selected = ak.from_parquet(parquet_file, columns=[("x", "y", "w.1")])
    assert array_selected.to_list() == [
        {
            "x": [
                {
                    "y": {
                        #  "z": [1, 2, 3],  Excluded
                        "w.1": 4,  # Selected
                    }
                }
            ]
        }
    ]

    ambig_array = ak.Array(
        [
            {
                "crazy": {
                    "dot": [11, 12, 13],
                },
                "crazy.dot": [21, 22, 23],
            }
        ]
    )
    parquet_file_ambig = os.path.join(tmp_path, "test_3088_array_ambig.parquet")
    ak.to_parquet(ambig_array, parquet_file_ambig)
    ambig_selected = ak.from_parquet(parquet_file_ambig, columns=[("crazy.dot",)])
    # Note: Currently, pyarrow.parquet cannot distinguish dots as separators
    # from dots as field names. It builds a dict of all possible indices,
    # and returns those. Even so, we still need the ability within Awkward to
    # disambiguate these two, which we now have. We would need further
    # feature work to create column name substitutions to work around this pyarrow
    # limitation should this be justified.
    assert ak.array_equal(ambig_selected, ambig_array)  # Slurped everything.
