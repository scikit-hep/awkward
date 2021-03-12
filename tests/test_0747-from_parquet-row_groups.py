# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import os

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


pytest.importorskip("pyarrow.parquet")


def test(tmp_path):
    filename = os.path.join(tmp_path, "test.parquet")
    ak.to_parquet(ak.repartition(range(8), 2), filename)

    assert ak.from_parquet(filename, row_groups=[1, 3]).tolist() == [2, 3, 6, 7]
    assert ak.from_parquet(filename, row_groups=[1, 3], lazy=True).tolist() == [
        2,
        3,
        6,
        7,
    ]

    assert ak.from_parquet(tmp_path, row_groups=[1, 3]).tolist() == [2, 3, 6, 7]
    assert ak.from_parquet(tmp_path, row_groups=[1, 3], lazy=True).tolist() == [
        2,
        3,
        6,
        7,
    ]

    ak.to_parquet.dataset(tmp_path)

    assert ak.from_parquet(tmp_path, row_groups=[1, 3]).tolist() == [2, 3, 6, 7]
    assert ak.from_parquet(tmp_path, row_groups=[1, 3], lazy=True).tolist() == [
        2,
        3,
        6,
        7,
    ]
