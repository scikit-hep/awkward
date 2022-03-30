# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_two_arrays():

    array = ak._v2.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.0, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ]
    )
    ak_array_1 = array["x"]
    ak_array_2 = array["y"]

    # An 'awkward' namespace will be added to the column name
    rdf = ak._v2.operations.convert.to_rdataframe({"x": ak_array_1, "y": ak_array_2})
    column_type = rdf.GetColumnType("awkward:x")

    done = compiler(
        """
        template <typename Array>
        struct MyFunctor {
            void operator()(Array  e) {
                cout << e.size() << endl;
            }
        };
        """
    )
    assert done is True

    try:
        rdf.Foreach(ROOT.MyFunctor[column_type], ["awkward:x"])
    except TypeError:
        print("FIXME: TypeError")


def test_one_array():

    array = ak._v2.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.0, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ]
    )
    ak_array_1 = array["x"]

    # An 'awkward' namespace will be added to the column name
    rdf = ak._v2.operations.convert.to_rdataframe({"xxx": ak_array_1})
    print(rdf)
