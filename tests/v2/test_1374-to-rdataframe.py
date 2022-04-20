# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_two_columns_as_rvecs():

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
    data_frame = ak._v2.operations.convert.to_rdataframe(
        {"x": ak_array_1, "y": ak_array_2}, flatlist_as_rvec=True
    )

    done = compiler(
        """
        template <typename Array>
        struct MyFunctorX {
            void operator()(const Array& a) {
                cout << "user function for X: " << a << endl;
            }
        };
        """
    )
    assert done is True

    done = compiler(
        """
        template <typename Array>
        struct MyFunctorY {
            void operator()(const Array& a) {
                cout << "user function for Y: ";
                for (int64_t i = 0; i < a.size(); i++) {
                    for (int64_t j = 0; j < a[i].size(); j++) {
                        cout << a[i][j] << ",";
                    }
                }
                cout << endl;
            }
        };
        """
    )
    assert done is True

    f_x = ROOT.MyFunctorX[data_frame.GetColumnType("awkward:x")]()
    f_y = ROOT.MyFunctorY[data_frame.GetColumnType("awkward:y")]()

    data_frame.Foreach(f_x, ["awkward:x"])
    data_frame.Foreach(f_y, ["awkward:y"])


def test_two_columns_as_vecs():

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
    data_frame = ak._v2.operations.convert.to_rdataframe(
        {"x": ak_array_1, "y": ak_array_2}
    )

    done = compiler(
        """
        template <typename Array>
        struct MyFunctorX_1 {
            void operator()(const Array& a) {
                cout << "user function for X: " << a << endl;
            }
        };
        """
    )
    assert done is True

    done = compiler(
        """
        template <typename Array>
        struct MyFunctorY_1 {
            void operator()(const Array& a) {
                cout << "user function for Y: ";
                for (int64_t i = 0; i < a.size(); i++) {
                    for (int64_t j = 0; j < a[i].size(); j++) {
                        cout << a[i][j] << ",";
                    }
                }
                cout << endl;
            }
        };
        """
    )
    assert done is True

    f_x = ROOT.MyFunctorX_1[data_frame.GetColumnType("awkward:x")]()
    f_y = ROOT.MyFunctorY_1[data_frame.GetColumnType("awkward:y")]()

    data_frame.Foreach(f_x, ["awkward:x"])
    data_frame.Foreach(f_y, ["awkward:y"])


def test_array():

    array = ak._v2.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.0, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ]
    )
    data_frame = ak._v2.operations.convert.to_rdataframe({"array": array})

    done = compiler(
        """
        template <typename Array>
        struct MyFunctor_1 {
            void operator()(const Array& a) {
                for (int64_t i = 0; i < a.size(); i++) {
                    cout << a[i].x() << ", ";
                    auto y = a[i].y();
                    for (int64_t j = 0; j < y.size(); j++) {
                        cout << y[j] << ", ";
                    }
                }
            }
        };
        """
    )
    assert done is True

    f = ROOT.MyFunctor_1[data_frame.GetColumnType("awkward:array")]()

    data_frame.Foreach(f, ["awkward:array"])
