# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._util import parse_memory_size

pytest.importorskip("pyarrow.parquet")

import pyarrow.parquet as pq


def test_parse_memory_size():
    # decimal and binary units, case-insensitive, flexible whitespace
    assert parse_memory_size("2 MB") == 2 * 1000**2
    assert parse_memory_size("  100mib ") == 100 * 1024**2
    # plain integers pass through unchanged
    assert parse_memory_size(100) == 100

    # everything else is rejected: bare numbers, unknown units, bool, float
    for bad in ("100", "100 furlongs", True, 1.5):
        with pytest.raises(TypeError):
            parse_memory_size(bad)


@pytest.mark.parametrize("iterative", [False, True])
def test_row_group_size_string(tmp_path, iterative):
    filename = tmp_path / "row-group-size-string.parquet"

    # 100000 rows of a single int16 column -> 2 bytes/row -> 200000 bytes
    array = ak.Array({"x": np.arange(100000, dtype=np.int16)})

    if iterative:
        ak.to_parquet_row_groups(iter((array,)), filename, row_group_size="50 KiB")
    else:
        ak.to_parquet(array, filename, row_group_size="50 KiB")

    metadata = pq.read_metadata(filename)

    # 50 KiB = 51200 bytes; at 2 bytes/row that is 25600 rows per group,
    # so 100000 rows -> 4 row groups (25600, 25600, 25600, 23200)
    assert metadata.num_row_groups == 4
    assert metadata.row_group(0).num_rows == 25600

    # data round-trips unchanged
    assert ak.to_list(ak.from_parquet(filename)["x"]) == ak.to_list(array["x"])


def test_row_group_size_string_empty(tmp_path):
    # an empty table has no rows/bytes to size against; fall back to the default
    filename = tmp_path / "empty.parquet"
    array = ak.Array({"x": np.array([], dtype=np.int64)})

    ak.to_parquet(array, filename, row_group_size="50 KiB")
    assert pq.read_metadata(filename).num_rows == 0
