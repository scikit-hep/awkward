# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import os

import numpy as np  # noqa: F401
import pytest

import awkward as ak

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_data_frame_integers(tmp_path):
    filename = os.path.join(tmp_path, "test-integers.root")

    ak_array_x = ak.Array([1, 2, 3, 4, 5])
    ak_array_y = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5])

    data_frame = ak.to_rdataframe({"x": ak_array_x, "y": ak_array_y})

    assert data_frame.GetColumnType("x") == "int64_t"
    assert data_frame.GetColumnType("y") == "double"

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x", "y"),
    )
    assert ak_array_x.to_list() == ak_array_out["x"].to_list()
    assert ak_array_y.to_list() == ak_array_out["y"].to_list()

    data_frame.Snapshot("Test", filename, ("x", "y"))


def test_data_frame_vec_of_vec_of_real(tmp_path):
    filename = os.path.join(tmp_path, "test-listarray.root")

    ak_array_in = ak.Array([[[1.1], [2.2]], [[3.3], [4.4, 5.5]]])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x").startswith("awkward::ListArray_")

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()

    with pytest.raises(SystemError):
        data_frame.Snapshot("ListArray", filename, ("x",))


def test_data_frame_rvec_filter(tmp_path):
    filename = os.path.join(tmp_path, "test-listarray2.root")

    ak_array_x = ak.Array([[1, 2], [3], [4, 5]])
    ak_array_y = ak.Array([[1.0, 1.1], [2.2, 3.3, 4.4], [5.5]])

    data_frame = ak.to_rdataframe({"x": ak_array_x, "y": ak_array_y})
    rdf3 = data_frame.Filter("x.size() >= 2")

    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<int64_t>"
    assert data_frame.GetColumnType("y") == "ROOT::VecOps::RVec<double>"

    ak_array_out = ak.from_rdataframe(
        rdf3,
        columns=(
            "x",
            "y",
        ),
    )
    assert ak_array_out["x"].to_list() == [[1, 2], [4, 5]]
    assert ak_array_out["y"].to_list() == [[1.0, 1.1], [5.5]]

    rdf4 = data_frame.Filter("y.size() == 2")
    ak_array_out = ak.from_rdataframe(
        rdf4,
        columns=(
            "x",
            "y",
        ),
    )
    assert ak_array_out["x"].to_list() == [[1, 2]]
    assert ak_array_out["y"].to_list() == [[1.0, 1.1]]

    data_frame.Snapshot(
        "ListArray",
        filename,
        (
            "x",
            "y",
        ),
    )
