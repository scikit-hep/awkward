# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_data_frame_integers():
    ak_array_x = ak.Array([1, 2, 3, 4, 5])
    ak_array_y = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5])

    data_frame = ak.to_rdataframe({"x": ak_array_x, "y": ak_array_y})

    assert data_frame.GetColumnType("x") == "int64_t"
    assert data_frame.GetColumnType("y") == "double"

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x", "y"),
    )
    assert ak_array_x.to_list() == ak_array_out["x"].to_list()
    assert ak_array_y.to_list() == ak_array_out["y"].to_list()


def test_data_frame_vec_of_vec_of_real():
    ak_array_in = ak.Array([[[1.1], [2.2]], [[3.3], [4.4, 5.5]]])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x").startswith("awkward::ListArray_")

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_data_frame_filter():
    ak_array_x = ak.Array([1, 2, 3, 4, 5])
    ak_array_y = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5])

    data_frame = ak.to_rdataframe({"x": ak_array_x, "y": ak_array_y})
    rdf3 = data_frame.Filter("x > 3")

    assert data_frame.GetColumnType("x") == "int64_t"
    assert data_frame.GetColumnType("y") == "double"

    ak_array_out = ak.from_rdataframe(
        rdf3,
        columns=(
            "x",
            "y",
        ),
    )
    assert ak_array_x[3:].to_list() == ak_array_out["x"].to_list()
    assert ak_array_y[3:].to_list() == ak_array_out["y"].to_list()


def test_data_frame_rvec_filter():
    ak_array_x = ak.Array([[1, 2], [3], [4, 5]])
    ak_array_y = ak.Array([[1.0, 1.1], [2.2, 3.3, 4.4], [5.5]])

    data_frame = ak.to_rdataframe({"x": ak_array_x, "y": ak_array_y})
    rdf3 = data_frame.Filter("x.size() >= 2")

    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<int64_t>"
    assert data_frame.GetColumnType("y") == "ROOT::VecOps::RVec<double>"

    ak_array_out = ak.from_rdataframe(
        rdf3,
        columns=(
            "x",
            "y",
        ),
    )
    assert ak_array_out["x"].to_list() == [[1, 2], [4, 5]]
    assert ak_array_out["y"].to_list() == [[1.0, 1.1], [5.5]]

    rdf4 = data_frame.Filter("y.size() == 2")
    ak_array_out = ak.from_rdataframe(
        rdf4,
        columns=(
            "x",
            "y",
        ),
    )
    assert ak_array_out["x"].to_list() == [[1, 2]]
    assert ak_array_out["y"].to_list() == [[1.0, 1.1]]


def test_data_frame_double():
    ak_array_in = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "double"

    ak_array_out = ak.from_rdataframe(data_frame, columns=("x",))
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_data_frame_vec_of_vec():
    array = ak.Array(
        [
            [
                {"x": 1.1, "y": [1]},
                {"x": None, "y": [1, 2]},
                {"x": 3.3, "y": [1, 2, 3]},
            ],
            [],
            [{"x": None, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}],
        ]
    )
    # ] * 10000)

    rdf2 = ak.to_rdataframe({"array": array})
    # We create a matrix RxC here
    # Note when dimensions R and C are large, the following code suffers
    # from potential performance penalties caused by frequent reallocation
    # of memory by the push_back() function. This should be used only when
    # vector dimensions are not known in advance.
    rdf3 = rdf2.Define(
        "output",
        """
    std::vector<std::vector<double>> tmp1;

    for (auto record : array) {
        std::vector<double> tmp2;
        for (auto number : record.y()) {
            tmp2.push_back(number * number);
        }
        tmp1.push_back(tmp2);
    }
    return tmp1;
    """,
    )

    assert rdf3.GetColumnType("output") == "vector<vector<double> >"

    rdf4 = rdf3.Define(
        "output2",
        """
    std::vector<std::vector<std::vector<double>>> tmp1;

    for (auto record : array) {
        std::vector<std::vector<double>> tmp2;
        // we can check if it's None:
        // if (record.x().has_value())
        // or set it to 1 so that we do not scale:
        double x_number = record.x().value_or(1);
        for (auto number : record.y()) {
            std::vector<double> tmp3;
            for (int64_t i = 0; i < std::rint(x_number); i++) {
                double value = x_number * number;
                tmp3.push_back(value);
            }
            tmp2.push_back(tmp3);
        }
        tmp1.push_back(tmp2);
    }
    return tmp1;
    """,
    )
    assert rdf4.GetColumnType("output2") == "vector<vector<vector<double> > >"

    out = ak.from_rdataframe(
        rdf4,
        columns=(
            "output",
            "output2",
        ),
    )

    assert out["output"].to_list() == (array["y"] * array["y"] * 1.0).to_list()
    result = ak.Array(
        [
            [
                [[1.1]],  # "x" is 1 - "y" values are unchanged, and each is nesed
                [
                    [1.0],
                    [2.0],
                ],  # "x" is None - "y" values are unchanged, and each is nesed
                [
                    [3.3, 3.3, 3.3],
                    [6.6, 6.6, 6.6],
                    [9.899999999999999, 9.899999999999999, 9.899999999999999],
                ],  # "x" is 3.3 - "y" values are scaled by 3.3 and each is nesed 3 times
            ],
            [],
            [
                [
                    [1.0],
                    [2.0],
                    [3.0],
                    [4.0],
                ],  # "x" is None - "y" values are unchanged, and each is nesed
                [
                    [5.5, 5.5, 5.5, 5.5, 5.5, 5.5],
                    [11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
                    [16.5, 16.5, 16.5, 16.5, 16.5, 16.5],
                    [22.0, 22.0, 22.0, 22.0, 22.0, 22.0],
                    [27.5, 27.5, 27.5, 27.5, 27.5, 27.5],
                ],  # "x" is 5.5 - "y" values are scaled by 5.5 and each is nesed 5 times
            ],
        ]
    )
    assert out["output2"].to_list() == result.to_list()


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
        columns=(
            "x",
            "y",
            "r",
        ),
    )

    assert array["x"].layout.form == ak.forms.ListOffsetForm(
        "i64", ak.forms.NumpyForm("float64")
    )
    assert array["y"].layout.form == ak.forms.ListOffsetForm(
        "i64", ak.forms.NumpyForm("float64")
    )
    assert array["r"].layout.form == ak.forms.ListOffsetForm(
        "i64", ak.forms.NumpyForm("float64")
    )

    assert array.layout.form == ak.forms.RecordForm(
        [
            ak.forms.ListOffsetForm("i64", ak.forms.NumpyForm("float64")),
            ak.forms.ListOffsetForm("i64", ak.forms.NumpyForm("float64")),
            ak.forms.ListOffsetForm("i64", ak.forms.NumpyForm("float64")),
        ],
        [
            "x",
            "y",
            "r",
        ],
    )
