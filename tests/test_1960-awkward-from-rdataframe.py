# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak
import awkward._connect.cling
import awkward._lookup

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_unknown_column_type():

    example1 = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5])
    data_frame = ak.to_rdataframe(
        {
            "one_float": example1,
        }
    )

    compiler(
        """
    struct TwoInts {
        int a, b;
    };

    template<typename T>
    ROOT::RDF::RNode MyTransformation_to_TwoInts(ROOT::RDF::RNode df) {
        auto myFunc = [](T x){ return TwoInts{(int)x, (int)2*x};};
        return df.Define("two_ints", myFunc, {"one_float"});
    }
    """
    )

    data_frame_transformed = ROOT.MyTransformation_to_TwoInts[  # noqa: F841
        data_frame.GetColumnType("one_float")
    ](ROOT.RDF.AsRNode(data_frame))

    with pytest.raises(TypeError, match=r"column's type"):
        ak.from_rdataframe(
            data_frame_transformed,
            columns=("two_ints",),
        )
