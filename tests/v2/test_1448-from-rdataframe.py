# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_data_frame():
    data_frame = ROOT.RDataFrame(10).Define("x", "gRandom->Rndm()")
    data_frame_xy = data_frame.Define("y", "x*2")

    ak_array_x = ak._v2.from_rdataframe(
        data_frame_xy, column="x", column_as_record=False
    )
    assert ak_array_x.layout.form == ak._v2.forms.NumpyForm("float64")

    ak_record_array_x = ak._v2.from_rdataframe(
        data_frame_xy, column="x", column_as_record=True
    )
    assert ak_record_array_x.layout.form == ak._v2.forms.RecordForm(
        [ak._v2.forms.NumpyForm("float64")], "x"
    )

    ak_record_array_y = ak._v2.from_rdataframe(
        data_frame_xy, column="y", column_as_record=True
    )
    ak_array = ak._v2.zip([ak_record_array_x, ak_record_array_y])
    assert ak_array.layout.form == ak._v2.forms.RecordForm(
        contents=[
            ak._v2.forms.RecordForm([ak._v2.forms.NumpyForm("float64")], "x"),
            ak._v2.forms.RecordForm([ak._v2.forms.NumpyForm("float64")], "y"),
        ],
        fields=None,
    )
