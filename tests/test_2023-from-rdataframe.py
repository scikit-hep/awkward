# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak
import awkward._connect.cling
import awkward._lookup

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_rdf_column_of_Long64_t_type():
    data_frame = ROOT.RDataFrame(10).Define("x", "(Long64_t)(gRandom->Rndm()*1000.)")
    assert data_frame.GetColumnType("x") == "Long64_t"

    ak_array = ak.from_rdataframe(
        data_frame,
        columns="x",
    )
    assert ak_array.layout.form == ak.forms.NumpyForm("int64")
