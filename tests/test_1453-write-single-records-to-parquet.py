# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import os

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401

pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.parquet")
pytest.importorskip("fsspec")


def test(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")

    original = ak.Record({"x": 1, "y": [1, 2, 3], "z": "THREE"})

    assert ak.from_arrow(ak.to_arrow(original)).tolist() == original.tolist()

    assert ak.from_arrow(ak.to_arrow_table(original)).tolist() == original.tolist()

    ak.to_parquet(original, filename)
    assert ak.from_parquet(filename).tolist() == original.tolist()
