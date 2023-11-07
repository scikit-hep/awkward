# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np  # noqa: F401
import pytest

import awkward as ak

ROOT = pytest.importorskip("ROOT")

ROOT.ROOT.EnableImplicitMT(1)

compiler = ROOT.gInterpreter.Declare


def test_data_frame_filter():
    array_x = ak.Array(
        [
            {"x": [1.1, 1.2, 1.3]},
            {"x": [2.1, 2.2]},
            {"x": [3.1]},
            {"x": [4.1, 4.2, 4.3, 4.4]},
            {"x": [5.1]},
        ]
    )
    array_y = ak.Array([1, 2, 3, 4, 5])
    array_z = ak.Array([[1.1], [2.1, 2.3, 2.4], [3.1], [4.1, 4.2, 4.3], [5.1]])

    df = ak.to_rdataframe({"x": array_x, "y": array_y, "z": array_z})

    assert str(df.GetColumnType("x")).startswith("awkward::Record_")
    assert df.GetColumnType("y") == "int64_t"
    assert df.GetColumnType("z") == "ROOT::VecOps::RVec<double>"

    df = df.Filter("y % 2 == 0")

    out = ak.from_rdataframe(
        df,
        columns=(
            "x",
            "y",
            "z",
        ),
    )
    assert out["x"].tolist() == [{"x": [2.1, 2.2]}, {"x": [4.1, 4.2, 4.3, 4.4]}]
    assert out["y"].tolist() == [2, 4]
    assert out["z"].tolist() == [[2.1, 2.3, 2.4], [4.1, 4.2, 4.3]]
    assert len(out) == 2
