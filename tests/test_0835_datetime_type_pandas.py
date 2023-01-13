# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak

pandas = pytest.importorskip("pandas")

to_list = ak.operations.to_list


def test_from_pandas():
    values = {"time": ["20190902093000", "20190913093000", "20190921200000"]}
    df = pandas.DataFrame(values, columns=["time"])
    df["time"] = pandas.to_datetime(df["time"], format="%Y%m%d%H%M%S")
    array = ak.contents.NumpyArray(df["time"].values)
    assert to_list(array) == df["time"].values.tolist()
    array2 = ak.highlevel.Array(df["time"].values)
    assert to_list(array2) == df["time"].values.tolist()
