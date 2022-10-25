# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401
import awkward._connect.cling  # noqa: E402
import awkward._lookup  # noqa: E402

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
