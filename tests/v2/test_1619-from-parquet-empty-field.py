# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import os

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pytest.importorskip("pyarrow.parquet")


def test_no_extension(tmp_path):
    array = ak._v2.Array(
        [
            [
                {"x": 1, "y": 1.1},
                {"x": 2, "y": 2.2},
                {"x": 3, "y": 3.3},
            ],
            [
                {"x": 1, "y": 1.1},
                {"x": 2, "y": 2.2},
            ],
        ]
    )
    path = os.path.join(tmp_path, "array-no-ext.parquet")

    ak._v2.to_parquet(array, path, extensionarray=False)

    result = ak._v2.from_parquet(path, columns=["x"])
    assert result.fields == ["x"]
