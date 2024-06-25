# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
# ruff: noqa: E402

from __future__ import annotations

import os

import pytest

import awkward as ak

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def test_parquet_subcolumn_select(tmp_path):
    ak_tbl = ak.Array(
        {
            "a": [
                {"lbl": "item 1", "idx": 11, "ids": [1, 2, 3]},
                {"lbl": "item 2", "idx": 12, "ids": [51, 52]},
                {"lbl": "item 3", "idx": 13, "ids": [61, 62, 63, 64]},
            ],
            "b": [
                [[111, 112], [121, 122]],
                [[211, 212], [221, 222]],
                [[311, 312], [321, 322]],
            ],
        }
    )
    parquet_file = os.path.join(tmp_path, "test_3514.parquet")
    ak.to_parquet(ak_tbl, parquet_file)

    selection = ak.from_parquet(parquet_file, columns=["a.ids", "b"])
    assert selection["a"].to_list() == [
        {"ids": [1, 2, 3]},
        {"ids": [51, 52]},
        {"ids": [61, 62, 63, 64]},
    ]
    assert selection["b"].to_list() == ak_tbl["b"].to_list()
