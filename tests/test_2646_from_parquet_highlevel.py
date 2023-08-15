# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest

import awkward as ak

pytest.importorskip("pyarrow")


def test(tmp_path):
    ak.to_parquet([1, 2, 3], tmp_path / "test.parquet", extensionarray=True)
    ak.to_parquet([1, 2, 3], tmp_path / "test2.parquet", extensionarray=True)

    # Single paths
    layout = ak.from_parquet(tmp_path / "test.parquet", highlevel=False)
    assert isinstance(layout, ak.contents.Content)

    # Multiple paths
    layout = ak.from_parquet(tmp_path / "test*.parquet", highlevel=False)
    assert isinstance(layout, ak.contents.Content)
