# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak

ROOT = pytest.importorskip("ROOT")


compiler = ROOT.gInterpreter.Declare


def test_data_frame_integers():
    ak_array_in = ak.Array([1, 2, 3, 4, 5])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "int64_t"

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_data_frame_double():
    ak_array_in = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "double"

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_data_frame_char():
    ak_array_in = ak.Array(["a", "b", "c", "d", "e"])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "std::string"

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_data_frame_complex():
    ak_array_in = ak.Array([1.1 + 0.1j, 2.2 + 0.2j, 3.3 + 0.3j, 4.4 + 0.4j, 5.5 + 0.5j])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "std::complex<double>"

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_data_frame_listoffset_integers():
    ak_array_in = ak.Array([[1], [2, 3, 4], [5]])

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<int64_t>"

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
    assert ak_array_in.to_list() == ak_array_out["x"].to_list()


def test_data_frame_listoffset_listoffset_double():
    ak_array_in = ak.Array(
        [
            [[1.1, 2.2, 3.3]],
            [[4.4, 5.5]],
            [[6.6], [7.7, 8.8, 9.9]],
        ]
    )

    data_frame = ak.to_rdataframe({"x": ak_array_in})

    # awkward::ListArray_ type
    # assert data_frame.GetColumnType("x") == "ROOT::VecOps::RVec<double>"

    ak_array_out = ak.from_rdataframe(
        data_frame,
        columns=("x",),
    )
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
    out = ak.from_rdataframe(
        rdf3,
        columns=("output",),
    )
    assert out["output"].to_list() == (array["y"] * array["y"] * 1.0).to_list()

    rdf3 = rdf2.Define(
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
    assert rdf3.GetColumnType("output2") == "vector<vector<vector<double> > >"
    out = ak.from_rdataframe(
        rdf3,
        columns=("output2",),
    )
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
