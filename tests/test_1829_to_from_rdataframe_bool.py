# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak
import awkward._connect.cling
import awkward._lookup

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_data_frame_boolean():
    ak_array_in = ak.Array([True, False, True])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "bool"

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_does_rdataframe_see_these_as_boolean():
    ak_array_in = ak.Array([True, False, True])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "bool"

    data_frame_2 = data_frame.Define("y", "!x")

    ak_array_out = ak.from_rdataframe(
        data_frame_2,
        columns=("y",),
    )
    assert [not x for x in ak_array_in] == ak_array_out["y"].to_list()


def test_filters_as_well():
    ak_array_in = ak.Array([True, False, True])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "bool"

    data_frame_2 = data_frame.Filter("x")

    ak_array_out = ak.from_rdataframe(
        data_frame_2,
        columns=("x",),
    )
    assert [True, True] == ak_array_out["x"].to_list()
