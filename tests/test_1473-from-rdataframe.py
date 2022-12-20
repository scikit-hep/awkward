# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak
import awkward._connect.cling
import awkward._lookup

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_to_from_data_frame_large():
    # Note, with n = 30 (14348907) this test takes ~40 sec to run on my laptop
    n = 6
    assert 2 * (n // 2) == n
    rows = 3 ** (n // 2)
    cols = n

    arr = np.zeros((rows, cols), dtype=np.int64)
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

    ak_array_in = ak.from_numpy(arr, regulararray=True)

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert len(ak_array_in) == len(ak_array_out)


@pytest.mark.skip(reason="FIXME: arrays of boolean are not supported yet")
def test_data_frame_boolean():
    ak_array_in = ak.Array([True, False, True, True, True])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "bool"

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out.to_list()


def test_data_frame_integers():
    ak_array_in = ak.Array([1, 2, 3, 4, 5])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "int64_t"

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_data_frame_real():
    ak_array_in = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "double"

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_data_frame_complex():
    ak_array_in = ak.Array([1.0 + 0.1j, 2.0 + 0.2j, 3.0 + 0.3j, 4.0 + 0.4j, 5.0 + 0.5j])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "std::complex<double>"

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_data_frame_strings():
    ak_array_in = ak.Array(["one", "two", "three"])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "std::string"

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_data_frame_vec_of_integers():
    ak_array_in = ak.Array([[1, 2], [3], [4, 5]])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<int64_t>"

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_data_frame_vec_of_real():
    ak_array_in = ak.Array([[1.1, 2.2], [3.3], [4.4, 5.5]])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<double>"

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_data_frame_vec_of_complex():
    ak_array_in = ak.Array(
        [[1.0 + 0.1j, 2.0 + 0.2j], [3.0 + 0.3j], [4.0 + 0.4j, 5.0 + 0.5j]]
    )

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<std::complex<double>>"

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_data_frame_vec_of_strings():
    ak_array_in = ak.Array([["one"], ["two", "three"]])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x").startswith("awkward::ListArray_")

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_data_frame_vec_of_vec_of_integers():
    ak_array_in = ak.Array([[[1], [2]], [[3], [4, 5]]])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x").startswith("awkward::ListArray_")

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_data_frame_vec_of_vec_of_real():
    ak_array_in = ak.Array([[[1.1], [2.2]], [[3.3], [4.4, 5.5]]])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x").startswith("awkward::ListArray_")

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_data_frame_vec_of_vec_of_complex():
    ak_array_in = ak.Array(
        [[[1.0 + 0.1j], [2.0 + 0.2j]], [[3.0 + 0.3j], [4.0 + 0.4j, 5.0 + 0.5j]]]
    )

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x").startswith("awkward::ListArray_")

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_rdata_frame_vecs_as_records():
    data_frame = ROOT.RDataFrame(10).Define("x", "gRandom->Rndm()")
    data_frame_xy = data_frame.Define("y", "x*2")

    ak_array_x = ak.from_rdataframe(
        data_frame_xy,
        columns=("x",),
    )
    assert ak_array_x["x"].layout.form == ak.forms.NumpyForm("float64")

    ak_record_array_x = ak.from_rdataframe(
        data_frame_xy,
        columns=("x",),
    )
    assert ak_record_array_x.layout.form == ak.forms.RecordForm(
        [ak.forms.NumpyForm("float64")], "x"
    )

    ak_record_array_y = ak.from_rdataframe(
        data_frame_xy,
        columns=("y",),
    )
    ak_array = ak.zip([ak_record_array_x, ak_record_array_y])
    assert ak_array.layout.form == ak.forms.RecordForm(
        contents=[
            ak.forms.RecordForm([ak.forms.NumpyForm("float64")], "x"),
            ak.forms.RecordForm([ak.forms.NumpyForm("float64")], "y"),
        ],
        fields=None,
    )


def test_rdata_frame_vecs_of_complex():
    data_frame = ROOT.RDataFrame(10).Define("x", "gRandom->Rndm()")
    data_frame_xy = data_frame.Define("y", "x*2 +1i")

    ak_array_y = ak.from_rdataframe(
        data_frame_xy,
        columns=("y",),
    )
    assert ak_array_y["y"].layout.form == ak.forms.NumpyForm("complex128")


def test_rdata_frame_rvecs_as_records():
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

    array = ak.from_rdataframe(
        data_frame_x_y_r,
        columns=("r",),
    )

    assert array.layout.form == ak.forms.RecordForm(
        [ak.forms.ListOffsetForm("i64", ak.forms.NumpyForm("float64"))], ["r"]
    )


def test_to_from_data_frame():
    ak_array_in = ak.Array([[1.1], [2.2, 3.3, 4.4], [5.5, 6.6]])
    assert ak_array_in.layout.content.is_contiguous is True

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<double>"

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_out["x"].layout.content.is_contiguous is True

    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_to_from_data_frame_rvec_of_rvec():
    ak_array_in = ak.Array([[[1.1]], [[2.2, 3.3], [4.4]], [[5.5, 6.6], []]])

    data_frame = ak.to_rdataframe({"x": ak_array_in})
    assert data_frame.GetColumnType("x").startswith("awkward::ListArray_")

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )

    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_to_from_data_frame_rvec_of_rvec_of_rvec():
    ak_array_in = ak.Array(
        [[[[1.1]]], [[[2.2], [3.3], [], [4.4]]], [[[], [5.5, 6.6], []]]]
    )

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )

    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_to_from_data_frame_columns_as_string():
    ak_array_in = ak.Array(
        [[[[1.1]]], [[[2.2], [3.3], [], [4.4]]], [[[], [5.5, 6.6], []]]]
    )

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns="x",
    )

    assert ak_array_in.to_list() == ak_array_out.to_list()
