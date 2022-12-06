# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak
import awkward._connect.cling
import awkward._lookup

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_unknown_column_type():
    # To mark a Python callable to be used from C++, you have to use the decorator
    # provided by PyROOT passing the C++ types of the input arguments and the return
    # value.
    @ROOT.Numba.Declare(["float", "int"], "float")
    def pypow(x, y):
        return x**y

    # The Python callable is now available from C++ in the Numba namespace.
    # For example, we can use it from the interpreter.
    ROOT.gInterpreter.ProcessLine('cout << "2^3 = " << Numba::pypow(2, 3) << endl;')

    # Or we can use the callable as well within a RDataFrame workflow.
    data = (
        ROOT.RDataFrame(4)
        .Define("x", "(float)rdfentry_")
        .Define("x_pow3", "Numba::pypow(x, 3)")
        .AsNumpy()
    )

    print("pypow({}, 3) = {}".format(data["x"], data["x_pow3"]))  # noqa: T201

    # ROOT uses the numba Python package to create C++ functions from python ones.
    # We support as input and return types of the callable fundamental types and
    # ROOT::RVec thereof. See the following callable computing the power of the
    # elements in an array.
    @ROOT.Numba.Declare(
        ["ROOT::VecOps::RVec<float>", "int"], "ROOT::VecOps::RVec<float>"
    )
    def pypowarray(x, y):
        return x**y

    ROOT.gInterpreter.ProcessLine(
        """
    ROOT::VecOps::RVec<float> x = {0, 1, 2, 3};
    cout << "pypowarray(" << x << ", 3) =  " << Numba::pypowarray(x, 3) << endl;
    """
    )

    # and now with RDataFrame
    s = (
        ROOT.RDataFrame(1)
        .Define("x", "ROOT::VecOps::RVec<float>{1,2,3}")
        .Define("x2", "Numba::pypowarray(x, 2)")
        .Sum("x2")
    )  # 1 + 4 + 9 == 14
    print("sum(pypowarray({ 1, 2, 3 }, 2)) = ", s.GetValue())  # noqa: T201


def test_unknown_column_type_1():

    # class AwesomeModel:
    #     def predict(self, x):
    #         return x[0] * x[1]
    #
    # model = AwesomeModel()
    #
    # @ROOT.Numba.Declare(['float', 'float'], 'float')
    # def predictModel(var1, var2):
    #     return model.predict([var1, var2])
    #
    # df = ROOT.ROOT.RDataFrame(10).Define("x", "Numba::predictModel(var1, var2)")
    # print(df.AsNumpy()) # noqa: T201

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
