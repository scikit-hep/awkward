# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_data_frame():
    data_frame = ROOT.RDataFrame(10).Define("x", "gRandom->Rndm()")

    done = compiler(
        """
        template<typename T>
        struct test_data_frame_MyFunctorX {
            void operator()(T const& x) {
                cout << "user function for X: " << x << endl;
            }
        };
        """
    )
    assert done is True

    print(data_frame.GetColumnType("x"))

    f_x = ROOT.test_data_frame_MyFunctorX[data_frame.GetColumnType("x")]()
    data_frame.Foreach(f_x, ["x"])
    data_frame_y = data_frame.Define("y", "x*2")

    ak_array = ak._v2.from_rdataframe(
        data_frame_y, columns=["x"], exclude=["y"], columns_as_records=False
    )
    print(ak_array)
