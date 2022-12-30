# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak
import awkward._connect.cling
import awkward._lookup

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_two_columns():
    array = ak.Array(
        [
            [{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.2, 0.2]}],
            [],
            [{"x": 3, "y": [3.0, 0.3, 3.3]}],
        ]
    )
    ak_array_1 = array["x"]
    ak_array_2 = array["y"]
    data_frame = ak.to_rdataframe(
        {"x": ak_array_1, "y": ak_array_2}, flatlist_as_rvec=True
    )
    assert set(data_frame.GetColumnNames()) == {"x", "y", "awkward_index_"}
    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<int64_t>"
    assert data_frame.GetColumnType("y").startswith("awkward::ListArray_")


def test_two_columns_as_rvecs():
    ak_array_1 = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5])
    ak_array_2 = ak.Array([{"x": 1.1}, {"x": 2.2}, {"x": 3.3}, {"x": 4.4}, {"x": 5.5}])

    data_frame = ak.to_rdataframe({"x": ak_array_1, "y": ak_array_2})
    assert set(data_frame.GetColumnNames()) == {"x", "y", "awkward_index_"}
    assert data_frame.GetColumnType("x") == "double"
    assert data_frame.GetColumnType("y").startswith("awkward::Record_")

    cpp_list_x = ", ".join(str(e) for e in ak_array_1.to_list())
    cpp_list_y = ", ".join(str(e) for e in ak_array_2.x.to_list())

    done = compiler(
        f"""
        int ix = 0;
        double x_val[5] = {{ {cpp_list_x} }};

        template<typename T>
        struct CheckX {{
            void operator()(T const& x) {{
                R__ASSERT(x == x_val[ix++]);
            }}
        }};

        int iy = 0;
        double y_val[5] = {{ {cpp_list_y} }};

        template<typename T>
        struct CheckY {{
            void operator()(T const& y) {{
                R__ASSERT(y.x() == y_val[iy++]);
            }}
        }};
        """
    )
    assert done is True

    f_x = ROOT.CheckX[data_frame.GetColumnType("x")]()
    f_y = ROOT.CheckY[data_frame.GetColumnType("y")]()

    data_frame.Foreach(f_x, ["x"])
    data_frame.Foreach(f_y, ["y"])


def test_list_array():
    ak_array = ak.Array([[1.1], [2.2, 3.3, 4.4], [5.5, 6.6]])

    data_frame = ak.to_rdataframe({"x": ak_array})

    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<double>"

    done = compiler(
        """
        int64_t row = 0;

        ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>> row_vals =
        {{ 1.1},
         { 2.2, 3.3, 4.4 },
         { 5.5, 6.6 }
        };

        template<typename T>
        struct MyFunctor_RVec {
            void operator()(T const& x) {
                for( int64_t j = 0; j < x.size(); j++ ) {
                    R__ASSERT(x[j] == row_vals[row][j]);
                }
                row++;
            }
        };
        """
    )
    assert done is True

    f_x = ROOT.MyFunctor_RVec[data_frame.GetColumnType("x")]()
    data_frame.Foreach(f_x, ["x"])


def test_two_columns_as_vecs():
    ak_array_1 = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5])
    ak_array_2 = ak.Array([{"x": 1.1}, {"x": 2.2}, {"x": 3.3}, {"x": 4.4}, {"x": 5.5}])

    data_frame = ak.operations.to_rdataframe(
        {"x": ak_array_1, "y": ak_array_2}, flatlist_as_rvec=False
    )
    assert set(data_frame.GetColumnNames()) == {"x", "y", "awkward_index_"}
    assert data_frame.GetColumnType("x") == "double"
    assert data_frame.GetColumnType("y").startswith("awkward::Record_")

    cpp_list_x = ", ".join(str(e) for e in ak_array_1.to_list())
    cpp_list_y = ", ".join(str(e) for e in ak_array_2.x.to_list())

    done = compiler(
        f"""
        int i_xv = 0;
        double xv_val[5] = {{ {cpp_list_x} }};

        template<typename T>
        struct MyFunctorX_Vec {{
            void operator()(T const& x) {{
                R__ASSERT(x == xv_val[i_xv++]);
            }}
        }};

        int i_yv = 0;
        double yv_val[5] = {{ {cpp_list_y} }};

        template<typename T>
        struct MyFunctorY_Vec {{
            void operator()(T const& y) {{
                R__ASSERT(y.x() == yv_val[i_yv++]);
            }}
        }};
        """
    )
    assert done is True

    f_x = ROOT.MyFunctorX_Vec[data_frame.GetColumnType("x")]()
    f_y = ROOT.MyFunctorY_Vec[data_frame.GetColumnType("y")]()

    data_frame.Foreach(f_x, ["x"])
    data_frame.Foreach(f_y, ["y"])


def test_two_columns_transform_filter():
    example1 = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5])
    example2 = ak.Array([{"x": 1.1}, {"x": 2.2}, {"x": 3.3}, {"x": 4.4}, {"x": 5.5}])

    data_frame = ak.to_rdataframe({"one": example1, "two": example2})
    assert set(data_frame.GetColumnNames()) == {"one", "two", "awkward_index_"}

    compiler(
        """
template<typename T>
ROOT::RDF::RNode MyTransformation(ROOT::RDF::RNode df) {
    auto myFunc = [](T x){ return -x;};
    return df.Define("neg_one", myFunc, {"one"});
}
"""
    )

    data_frame_transformed = ROOT.MyTransformation[data_frame.GetColumnType("one")](
        ROOT.RDF.AsRNode(data_frame)
    )
    assert set(data_frame_transformed.GetColumnNames()) == {
        "neg_one",
        "one",
        "two",
        "awkward_index_",
    }
    assert data_frame_transformed.Count().GetValue() == 5

    data_frame2 = data_frame.Filter("one > 2.5")
    data_frame2_transformed = ROOT.MyTransformation[data_frame.GetColumnType("one")](
        ROOT.RDF.AsRNode(data_frame2)
    )
    assert data_frame2_transformed.Count().GetValue() == 3


def test_jims_example1():
    array = ak.Array([{"x": 1.1}, {"x": 2.2}, {"x": 3.3}, {"x": 4.4}, {"x": 5.5}])
    data_frame = ak.to_rdataframe({"some_array": array})
    assert set(data_frame.GetColumnNames()) == {"some_array", "awkward_index_"}
    data_frame_y = data_frame.Define("y", "some_array.x()")
    assert set(data_frame_y.GetColumnNames()) == {"some_array", "y", "awkward_index_"}

    cpp_list = ", ".join(str(e) for e in array.x.to_list())

    done = compiler(
        f"""
        int i_y1 = 0;
        double y1_val[5] = {{ {cpp_list} }};

        template<typename T>
        struct MyFunctorY_1 {{
            void operator()(T const& y) {{
                R__ASSERT(y == y1_val[i_y1++]);
            }}
        }};
        """
    )
    assert done is True

    f_y = ROOT.MyFunctorY_1[data_frame_y.GetColumnType("y")]()

    data_frame_y.Foreach(f_y, ["y"])


def test_jims_example2():
    example1 = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5])
    example2 = ak.Array(
        [
            {"x": [1.1, 1.2, 1.3]},
            {"x": [2.2, 2.21]},
            {"x": [3.3]},
            {"x": [4.4, 4.41, 4.42, 4.44]},
            {"x": [5.5]},
        ]
    )

    data_frame = ak.operations.to_rdataframe({"one": example1, "two": example2})

    assert data_frame.GetColumnType("one") == "double"
    assert data_frame.GetColumnType("two").startswith("awkward::Record_")

    done = compiler(
        """
        int i_one = 0;
        double one_val[5] = {1.1, 2.2, 3.3, 4.4, 5.5};

        template<typename T>
        struct check_one {
            void operator()(T const& one) {
                R__ASSERT(one == one_val[i_one++]);
            }
        };

        int i_two = 0;
        ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>> two_val =
            {{ 1.1, 1.2, 1.3 },
             { 2.2, 2.21 },
             { 3.3 },
             { 4.4, 4.41, 4.42, 4.44 },
             { 5.5 }
            };

        template<typename T>
        struct check_two {
            void operator()(T const& two) {
                for( int64_t j = 0; j < two.x().size(); j++ ) {
                    R__ASSERT(two.x()[j] == two_val[i_two][j]);
                }
                i_two++;
            }
        };
        """
    )
    assert done is True

    f_one = ROOT.check_one[data_frame.GetColumnType("one")]()
    f_two = ROOT.check_two[data_frame.GetColumnType("two")]()

    data_frame.Foreach(f_one, ["one"])
    data_frame.Foreach(f_two, ["two"])


def test_empty_array():
    array = ak.Array([])
    data_frame = ak.to_rdataframe({"empty_array": array})
    assert data_frame.GetColumnType("empty_array") == "double"
    assert data_frame.Count().GetValue() == 0


def test_empty_list_array():
    array = ak.Array([[], [], []])
    data_frame = ak.to_rdataframe({"empty_list_array": array})
    assert data_frame.GetColumnType("empty_list_array") == "ROOT::VecOps::RVec<double>"
    assert data_frame.Count().GetValue() == 3
