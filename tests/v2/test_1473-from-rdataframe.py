# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import awkward._v2._lookup  # noqa: E402
import awkward._v2._connect.cling  # noqa: E402

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_to_from_data_frame_large():
    n = 6  # 30
    assert 2 * (n // 2) == n
    rows = 3 ** (n // 2)
    cols = n

    arr = np.zeros((rows, cols), dtype=int)
    shape = (rows,)

    source = np.array([-1, 0, 1], dtype=np.int64)[:, None]

    for col in range(n // 2):
        shape = (
            -1,
            3,
            shape[-1] // 3,
        )
        col_view = arr[:, col]
        col_view.shape = shape
        col_view[:] = source

    ak_array_in = ak._v2.from_numpy(arr, regulararray=True)

    array = ak._v2.Array([ak_array_in])

    data_frame = ak._v2.to_rdataframe({"x": array})
    done = compiler(
        """
    template<typename T>
    struct MyFunctor_x {
        ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>> operator()(T x) {
            ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>> result;
            for (int64_t i = 0; i < x.size(); i++) {
                ROOT::VecOps::RVec<double> nested_result;
                for(int64_t j = 0; j < x[i].size(); j++) {
                    nested_result.emplace_back((double)x[i][j]);
                }
                result.emplace_back(nested_result);
            }
            return result;
        }
    };
    """
    )
    assert done is True

    f_x = ROOT.MyFunctor_x[data_frame.GetColumnType("x")]()

    data_frame_y = data_frame.Define("y", f_x, ["x"])

    ak_array_out = ak._v2.from_rdataframe(
        data_frame_y, column="y", column_as_record=False
    )
    assert array.to_list() == ak_array_out.to_list()


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


@pytest.mark.skip(reason="FIXME: arrays of boolean are not supported yet")
def test_data_frame_boolean():
    ak_array_in = ak._v2.Array([True, False, True, True, True])

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "bool"

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )
    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_data_frame_integers():
    ak_array_in = ak._v2.Array([1, 2, 3, 4, 5])

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "int64_t"

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )
    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_data_frame_real():
    ak_array_in = ak._v2.Array([1.1, 2.2, 3.3, 4.4, 5.5])

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "double"

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )
    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_data_frame_complex():
    ak_array_in = ak._v2.Array(
        [1.0 + 0.1j, 2.0 + 0.2j, 3.0 + 0.3j, 4.0 + 0.4j, 5.0 + 0.5j]
    )

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "std::complex<double>"

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )
    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_data_frame_strings():
    ak_array_in = ak._v2.Array(["one", "two", "three"])

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "std::string"

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )
    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_data_frame_vec_of_integers():
    ak_array_in = ak._v2.Array([[1, 2], [3], [4, 5]])

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<int64_t>"

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )
    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_data_frame_vec_of_real():
    ak_array_in = ak._v2.Array([[1.1, 2.2], [3.3], [4.4, 5.5]])

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<double>"

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )
    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_data_frame_vec_of_complex():
    ak_array_in = ak._v2.Array(
        [[1.0 + 0.1j, 2.0 + 0.2j], [3.0 + 0.3j], [4.0 + 0.4j, 5.0 + 0.5j]]
    )

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<std::complex<double>>"

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )
    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_data_frame_vec_of_strings():
    ak_array_in = ak._v2.Array([["one"], ["two", "three"]])

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x").startswith("awkward::ListArray_")

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )
    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_data_frame_vec_of_vec_of_integers():
    ak_array_in = ak._v2.Array([[[1], [2]], [[3], [4, 5]]])

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x").startswith("awkward::ListArray_")

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )
    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_data_frame_vec_of_vec_of_real():
    ak_array_in = ak._v2.Array([[[1.1], [2.2]], [[3.3], [4.4, 5.5]]])

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x").startswith("awkward::ListArray_")

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )
    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_data_frame_vec_of_vec_of_complex():
    ak_array_in = ak._v2.Array(
        [[[1.0 + 0.1j], [2.0 + 0.2j]], [[3.0 + 0.3j], [4.0 + 0.4j, 5.0 + 0.5j]]]
    )

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x").startswith("awkward::ListArray_")

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )
    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_data_frame_rvecs():
    data_frame = ROOT.RDataFrame(1024)
    coordDefineCode = """ROOT::VecOps::RVec<double> {0}(len);
                     std::transform({0}.begin(), {0}.end(), {0}.begin(), [](double){{return gRandom->Uniform(-1.0, 1.0);}});
                     return {0};"""

    data_frame_x_y = (
        data_frame.Define("len", "gRandom->Uniform(0, 16)")
        .Define("x", coordDefineCode.format("x"))
        .Define("y", coordDefineCode.format("y"))
    )

    # Now we have in hands d, a RDataFrame with two columns, x and y, which
    # hold collections of coordinates. The size of these collections vary.
    # Let's now define radii out of x and y. We'll do it treating the collections
    # stored in the columns without looping on the individual elements.
    data_frame_x_y_r = data_frame_x_y.Define("r", "sqrt(x*x + y*y)")
    assert data_frame_x_y_r.GetColumnType("r") == "ROOT::VecOps::RVec<double>"

    array = ak._v2.from_rdataframe(data_frame_x_y_r, column="r", column_as_record=True)

    assert array.layout.form == ak._v2.forms.RecordForm(
        [ak._v2.forms.ListOffsetForm("i64", ak._v2.forms.NumpyForm("float64"))], ["r"]
    )


def test_to_from_data_frame():
    ak_array_in = ak._v2.Array([[1.1], [2.2, 3.3, 4.4], [5.5, 6.6]])
    assert ak_array_in.layout.content.is_contiguous is True

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<double>"

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )
    assert ak_array_out.layout.content.is_contiguous is True

    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_to_from_data_frame_rvec_of_rvec():
    ak_array_in = ak._v2.Array([[[1.1]], [[2.2, 3.3], [4.4]], [[5.5, 6.6], []]])

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})
    assert data_frame.GetColumnType("x").startswith("awkward::ListArray_")

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )

    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_to_from_data_frame_rvec_of_rvec_of_rvec():
    ak_array_in = ak._v2.Array(
        [[[[1.1]]], [[[2.2], [3.3], [], [4.4]]], [[[], [5.5, 6.6], []]]]
    )

    data_frame = ak._v2.to_rdataframe({"x": ak_array_in})

    ak_array_out = ak._v2.from_rdataframe(
        data_frame, column="x", column_as_record=False
    )

    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_data_frame_complex_vecs():
    data_frame = ROOT.RDataFrame(10).Define("x", "gRandom->Rndm()")
    data_frame_xy = data_frame.Define("y", "x*2 +1i")

    ak_array_y = ak._v2.from_rdataframe(
        data_frame_xy, column="y", column_as_record=False
    )
    assert ak_array_y.layout.form == ak._v2.forms.NumpyForm("complex128")
