# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import os

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_no_fields(tmp_path):
    one = ak.Array([[1, 2, 3], [], [4, 5]])
    two = ak.Array([[6], [7, 8, 9, 10]])

    ak.to_parquet(one, tmp_path / "file1.parquet")
    ak.to_parquet(two, tmp_path / "file2.parquet")
    assert not os.path.exists(tmp_path / "_common_metadata")
    assert not os.path.exists(tmp_path / "_metadata")

    no_metadata = ak.from_parquet(tmp_path)
    assert no_metadata.tolist() == [[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]]

    no_metadata_lazy = ak.from_parquet(tmp_path, lazy=True)
    assert no_metadata_lazy.tolist() == [[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]]

    ak.to_parquet.dataset(tmp_path)
    assert os.path.exists(tmp_path / "_common_metadata")
    assert os.path.exists(tmp_path / "_metadata")

    with_metadata = ak.from_parquet(tmp_path)
    assert with_metadata.tolist() == [[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]]

    with_metadata_lazy = ak.from_parquet(tmp_path, lazy=True)
    assert with_metadata_lazy.tolist() == [[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]]


def test_with_fields(tmp_path):
    one_list = [[{"x": 1}, {"x": 2}, {"x": 3}], [], [{"x": 4}, {"x": 5}]]
    two_list = [[{"x": 6}], [{"x": 7}, {"x": 8}, {"x": 9}, {"x": 10}]]
    one = ak.Array(one_list)
    two = ak.Array(two_list)

    ak.to_parquet(one, tmp_path / "file1.parquet")
    ak.to_parquet(two, tmp_path / "file2.parquet")
    assert not os.path.exists(tmp_path / "_common_metadata")
    assert not os.path.exists(tmp_path / "_metadata")

    no_metadata = ak.from_parquet(tmp_path)
    assert no_metadata.tolist() == one_list + two_list

    no_metadata_lazy = ak.from_parquet(tmp_path, lazy=True)
    assert no_metadata_lazy.tolist() == one_list + two_list

    ak.to_parquet.dataset(tmp_path)
    assert os.path.exists(tmp_path / "_common_metadata")
    assert os.path.exists(tmp_path / "_metadata")

    with_metadata = ak.from_parquet(tmp_path)
    assert with_metadata.tolist() == one_list + two_list

    with_metadata_lazy = ak.from_parquet(tmp_path, lazy=True)
    assert with_metadata_lazy.tolist() == one_list + two_list
