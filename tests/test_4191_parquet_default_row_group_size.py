# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

pytest.importorskip("pyarrow.parquet")


@pytest.mark.parametrize("iterative", [False, True])
def test_default_row_group_size(tmp_path, iterative):
    filename = tmp_path / "default-row-group-size.parquet"
    array = ak.Array(np.zeros(1024 * 1024 + 1, dtype=np.int8))

    if iterative:
        metadata = ak.to_parquet_row_groups(iter((array,)), filename)
    else:
        metadata = ak.to_parquet(array, filename)

    assert metadata.num_row_groups == 2
