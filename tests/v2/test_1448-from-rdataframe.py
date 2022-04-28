# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_data_frame_vecs():
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


@pytest.mark.skip(reason="FIXME: nested types have not been implemented yet")
def test_data_frame_rvecs():
    data_frame = ROOT.RDataFrame(1024)
    coordDefineCode = """ROOT::VecOps::RVec<double> {0}(len);
                     std::transform({0}.begin(), {0}.end(), {0}.begin(), [](double){{return gRandom->Uniform(-1.0, 1.0);}});
                     return {0};"""

    d = (
        data_frame.Define("len", "gRandom->Uniform(0, 16)")
        .Define("x", coordDefineCode.format("x"))
        .Define("y", coordDefineCode.format("y"))
    )

    # Now we have in hands d, a RDataFrame with two columns, x and y, which
    # hold collections of coordinates. The size of these collections vary.
    # Let's now define radii out of x and y. We'll do it treating the collections
    # stored in the columns without looping on the individual elements.
    d1 = d.Define("r", "sqrt(x*x + y*y)")

    array = ak._v2.from_rdataframe(d1, column="r", column_as_record=True)
    print(array)
