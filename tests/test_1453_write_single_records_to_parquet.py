# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import os

import numpy as np  # noqa: F401
import pytest

import awkward as ak

pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.parquet")


def test(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")

    original = ak.Record({"x": 1, "y": [1, 2, 3], "z": "THREE"})

    assert ak.from_arrow(ak.to_arrow(original)).to_list() == original.to_list()

    assert ak.from_arrow(ak.to_arrow_table(original)).to_list() == original.to_list()

    ak.to_parquet(original, filename)
    assert ak.from_parquet(filename).to_list() == original.to_list()
