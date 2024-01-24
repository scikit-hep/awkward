from __future__ import annotations

import math
import os

import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")
uproot = pytest.importorskip("uproot")
skhep_testdata = pytest.importorskip("skhep_testdata")


def HZZ_test_100(tmp_file):
    batch_size = 100
    iterator = uproot.iterate(
        uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"],
        step_size=batch_size,
    )

    data = uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"].arrays()

    metadata = ak.to_parquet_row_groups(iterator, tmp_file)
    test = ak.from_parquet(tmp_file)

    assert len(test) == len(data)

    for name in test.fields:
        assert ak.all(test[name] == data[name])

    # Check row_group size with batch_size to see if writing in correct batch
    # sizes and ending up with right number of row_groups
    row_groups = metadata.num_row_groups
    num_rows = metadata.num_rows

    assert int(math.ceil(num_rows / batch_size)) == row_groups

    os.remove(tmp_file)

def HZZ_test_1000(tmp_file):
    batch_size = 1000
    iterator = uproot.iterate(
        uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"],
        step_size=batch_size,
    )

    data = uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"].arrays()

    metadata = ak.to_parquet_row_groups(iterator, tmp_file)
    test = ak.from_parquet(tmp_file)

    assert len(test) == len(data)

    for name in test.fields:
        assert ak.all(test[name] == data[name])

    # Check row_group size with batch_size to see if writing in correct batch
    # sizes and ending up with right number of row_groups
    row_groups = metadata.num_row_groups
    num_rows = metadata.num_rows

    assert int(math.ceil(num_rows / batch_size)) == row_groups

    os.remove(tmp_file)

def HZZ_test_100_MB(tmp_file):
    batch_size = "100 MB"
    iterator = uproot.iterate(
        uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"],
        step_size=batch_size,
    )

    data = uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"].arrays()

    metadata = ak.to_parquet_row_groups(iterator, tmp_file)
    test = ak.from_parquet(tmp_file)

    assert len(test) == len(data)

    for name in test.fields:
        assert ak.all(test[name] == data[name])

    # Check row_group size with batch_size to see if writing in correct batch
    # sizes and ending up with right number of row_groups
    row_groups = metadata.num_row_groups
    
    assert row_groups == 1
    
    os.remove(tmp_file)

def simple_test_params(tmp_file):
    batch_size = 1000
    arr = ak.Array(
        [
            [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
            [],
            [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}],
        ]
    )
    iterator = (i for i in arr)

    metadata = ak.to_parquet_row_groups(iterator, tmp_file)
    test = ak.from_parquet(tmp_file)
    
    print(len(arr[0]))
    print(test)
    assert len(test) == len(arr)

    for name in test.fields:
        assert ak.all(test[name] == arr[name])

    # Check row_group size with batch_size to see if writing in correct batch
    # sizes and ending up with right number of row_groups
    row_groups = metadata.num_row_groups
    num_rows = metadata.num_rows

    assert int(math.ceil(num_rows / batch_size)) == row_groups

    os.remove(tmp_file)

simple_test_params("path.parquet")