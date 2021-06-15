# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import os

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


pyarrow = pytest.importorskip("pyarrow")


def test_issue1():
    null_array = ak.Array({"null_col": [None]})
    tpe = ak.to_arrow_table(null_array).to_batches()[0].field("null_col").type
    assert tpe == pyarrow.null()


pyarrow_parquet = pytest.importorskip("pyarrow.parquet")


def test_issue2(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")

    null_table = pyarrow.Table.from_pydict({"null_col": pyarrow.array([None])})
    pyarrow_parquet.write_table(null_table, filename)

    assert ak.from_parquet(filename).type == ak.from_parquet(filename, lazy=True).type
