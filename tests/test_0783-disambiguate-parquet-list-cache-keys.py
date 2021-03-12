# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


pytest.importorskip("pyarrow.parquet")


def test(tmp_path):
    one = ak.Array([[], [{"x": [{"y": 1}]}]])
    two = ak.Array([[{"x": []}, {"x": [{"y": 1}]}]])
    three = ak.Array([[{"x": [{"y": 1}]}], [], [{"x": [{"y": 2}]}]])

    ak.to_parquet(one, tmp_path / "one.parquet")
    ak.to_parquet(two, tmp_path / "two.parquet")
    ak.to_parquet(three, tmp_path / "three.parquet")

    lazy_one = ak.from_parquet(tmp_path / "one.parquet", lazy=True)
    lazy_two = ak.from_parquet(tmp_path / "two.parquet", lazy=True)
    lazy_three = ak.from_parquet(tmp_path / "three.parquet", lazy=True)

    assert lazy_one.tolist() == [[], [{"x": [{"y": 1}]}]]
    assert lazy_two.tolist() == [[{"x": []}, {"x": [{"y": 1}]}]]
    assert lazy_three.tolist() == [[{"x": [{"y": 1}]}], [], [{"x": [{"y": 2}]}]]
