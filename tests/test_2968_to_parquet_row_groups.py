from __future__ import annotations

import math
import os

import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")
uproot = pytest.importorskip("uproot")
skhep_testdata = pytest.importorskip("skhep_testdata")


def HZZ_test(tmp_file, batch_size):
    iterator = uproot.iterate(
        uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"],
        step_size=batch_size,
    )

    data = uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"].arrays()

    ak.to_parquet_row_groups(iterator, tmp_file, row_group_size=batch_size)
    test = ak.from_parquet(tmp_file)

    assert len(test) == len(data)

    for name in test.fields:
        assert ak.all(test[name] == data[name])

    # Check row_group size with batch_size to see if writing in correct batch
    # sizes and ending up with right number of row_groups
    row_groups = pyarrow.parquet.read_metadata(tmp_file).num_row_groups
    num_rows = pyarrow.parquet.read_metadata(tmp_file).num_rows

    assert int(math.ceil(num_rows / batch_size)) == row_groups

    os.remove(tmp_file)
