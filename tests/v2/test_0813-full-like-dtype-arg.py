# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = ak._v2.Array(
        [
            [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}],
            [],
            [
                {"x": 3.3, "y": [1, 2, None, 3]},
                False,
                False,
                True,
                {"x": 4.4, "y": [1, 2, None, 3, 4]},
            ],
        ]
    )

    def assert_array_type(new_array, intended_type):
        """helper function to check each part of the array took the intended type"""
        assert isinstance(new_array[0][0]["x"], intended_type)
        assert isinstance(new_array[0][1]["x"], intended_type)
        assert isinstance(new_array[0][1]["y"][0], intended_type)
        assert isinstance(new_array[0][2]["x"], intended_type)
        assert isinstance(new_array[0][2]["y"][0], intended_type)
        assert isinstance(new_array[0][2]["y"][1], intended_type)
        assert isinstance(new_array[2][0]["x"], intended_type)
        assert isinstance(new_array[2][0]["y"][0], intended_type)
        assert isinstance(new_array[2][0]["y"][1], intended_type)
        assert isinstance(new_array[2][0]["y"][3], intended_type)
        assert isinstance(new_array[2][1], intended_type)
        assert isinstance(new_array[2][2], intended_type)
        assert isinstance(new_array[2][3], intended_type)
        assert isinstance(new_array[2][4]["x"], intended_type)
        assert isinstance(new_array[2][4]["y"][0], intended_type)
        assert isinstance(new_array[2][4]["y"][1], intended_type)
        assert isinstance(new_array[2][4]["y"][3], intended_type)
        assert isinstance(new_array[2][4]["y"][4], intended_type)

    int_type32 = ak._v2.operations.full_like(array, 12, dtype="int32")
    int_type64 = ak._v2.operations.full_like(array, 12, dtype="int64")
    float_type = ak._v2.operations.full_like(array, 12, dtype=float)

    assert int_type64.tolist() == float_type.tolist()
    assert_array_type(int_type32, np.int32)
    assert_array_type(int_type64, np.int64)
    assert_array_type(float_type, float)

    bool_type = ak._v2.operations.full_like(array, 12, dtype=bool)
    assert_array_type(bool_type, np.bool_)

    int_type32 = ak._v2.operations.full_like(array, -1.2, dtype="int32")
    int_type64 = ak._v2.operations.full_like(array, -1.2, dtype="int64")
    float_type = ak._v2.operations.full_like(array, -1.2, dtype=float)
    bool_type = ak._v2.operations.full_like(array, -1.2, dtype=bool)

    assert_array_type(int_type32, np.int32)
    assert_array_type(int_type64, np.int64)
    assert_array_type(float_type, float)
    assert_array_type(bool_type, np.bool_)

    int_type32 = ak._v2.operations.zeros_like(array, dtype="int32")
    int_type64 = ak._v2.operations.zeros_like(array, dtype="int64")
    float_type = ak._v2.operations.zeros_like(array, dtype=float)
    bool_type = ak._v2.operations.zeros_like(array, dtype=bool)

    assert int_type64.tolist() == float_type.tolist()
    assert int_type64.tolist() == bool_type.tolist()
    assert_array_type(int_type32, np.int32)
    assert_array_type(int_type64, np.int64)
    assert_array_type(float_type, float)
    assert_array_type(bool_type, np.bool_)

    int_type32 = ak._v2.operations.ones_like(array, dtype="int32")
    int_type64 = ak._v2.operations.ones_like(array, dtype="int64")
    float_type = ak._v2.operations.ones_like(array, dtype=float)
    bool_type = ak._v2.operations.ones_like(array, dtype=bool)

    assert int_type64.tolist() == float_type.tolist()
    assert int_type64.tolist() == bool_type.tolist()
    assert_array_type(int_type32, np.int32)
    assert_array_type(int_type64, np.int64)
    assert_array_type(float_type, float)
    assert_array_type(bool_type, np.bool_)

    array = ak.Array([["one", "two", "three"], [], ["four", "five"]])

    def assert_array_type(new_array, intended_type):
        """helper function to check each part of the array took the intended type"""
        assert isinstance(new_array[0][0], intended_type)
        assert isinstance(new_array[0][1], intended_type)
        assert isinstance(new_array[0][2], intended_type)
        assert isinstance(new_array[2][0], intended_type)
        assert isinstance(new_array[2][1], intended_type)

    int_type32 = ak._v2.operations.full_like(array, 12, dtype="int32")
    int_type64 = ak._v2.operations.full_like(array, 12, dtype="int64")
    float_type = ak._v2.operations.full_like(array, 12, dtype=float)

    assert int_type64.tolist() == float_type.tolist()
    assert_array_type(int_type32, np.int32)
    assert_array_type(int_type64, np.int64)
    assert_array_type(float_type, float)

    bool_type = ak._v2.operations.full_like(array, 12, dtype=bool)
    assert_array_type(bool_type, np.bool_)

    int_type32 = ak._v2.operations.full_like(array, -1.2, dtype="int32")
    int_type64 = ak._v2.operations.full_like(array, -1.2, dtype="int64")
    float_type = ak._v2.operations.full_like(array, -1.2, dtype=float)
    bool_type = ak._v2.operations.full_like(array, -1.2, dtype=bool)

    assert_array_type(int_type32, np.int32)
    assert_array_type(int_type64, np.int64)
    assert_array_type(float_type, float)
    assert_array_type(bool_type, np.bool_)

    int_type32 = ak._v2.operations.zeros_like(array, dtype="int32")
    int_type64 = ak._v2.operations.zeros_like(array, dtype="int64")
    float_type = ak._v2.operations.zeros_like(array, dtype=float)
    bool_type = ak._v2.operations.zeros_like(array, dtype=bool)

    assert int_type64.tolist() == float_type.tolist()
    assert int_type64.tolist() == bool_type.tolist()
    assert_array_type(int_type32, np.int32)
    assert_array_type(int_type64, np.int64)
    assert_array_type(float_type, float)
    assert_array_type(bool_type, np.bool_)

    int_type32 = ak._v2.operations.ones_like(array, dtype="int32")
    int_type64 = ak._v2.operations.ones_like(array, dtype="int64")
    float_type = ak._v2.operations.ones_like(array, dtype=float)
    bool_type = ak._v2.operations.ones_like(array, dtype=bool)
    assert int_type64.tolist() == float_type.tolist()
    assert int_type64.tolist() == bool_type.tolist()
    assert_array_type(int_type32, np.int32)
    assert_array_type(int_type64, np.int64)
    assert_array_type(float_type, float)
    assert_array_type(bool_type, np.bool_)
