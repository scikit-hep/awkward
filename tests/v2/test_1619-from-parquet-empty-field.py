# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import os

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


pytest.importorskip("pyarrow.parquet")


def test(tmp_path):
    array = ak._v2.Array(
        [
            {"": {"x": 1, "y": 1.1}},
            {"": {"x": 2, "y": 2.2}},
            {"": {"x": 3, "y": 3.3}},
            {"": {"x": 4, "y": 4.4}},
            {"": {"x": 5, "y": 5.5}},
            {"": {"x": 6, "y": 6.6}},
            {"": {"x": 7, "y": 7.7}},
            {"": {"x": 8, "y": 8.8}},
            {"": {"x": 9, "y": 9.9}},
        ]
    )
    path = os.path.join(tmp_path, "array3.parquet")

    ak._v2.to_parquet(array, path)

    result = ak._v2.from_parquet(path, columns=["x"])
    assert result.fields == ["x"]
