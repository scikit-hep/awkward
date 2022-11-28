# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import copy

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test():
    np_array = np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    one = ak.Array(np_array)

    np_array[1] = 999
    assert to_list(one) == [0.0, 999, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

    two = copy.copy(one)
    np_array[3] = 123
    assert to_list(two) == [0.0, 999, 2.2, 123, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

    three = copy.deepcopy(two)
    four = np.copy(two)
    five = ak.operations.copy(two)
    np_array[5] = 321
    assert to_list(three) == [0.0, 999, 2.2, 123, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    assert to_list(four) == [0.0, 999, 2.2, 123, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    assert to_list(five) == [0.0, 999, 2.2, 123, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

    assert to_list(copy.deepcopy(ak.Array([[1, 2, 3], [], [4, 5]]))) == to_list(
        ak.operations.copy(ak.Array([[1, 2, 3], [], [4, 5]]))
    )
    assert to_list(copy.deepcopy(ak.Record({"one": 1, "two": 2.2}))) == to_list(
        ak.operations.copy(ak.Record({"one": 1, "two": 2.2}))
    )

    underlying_array = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    wrapper = ak.Array(underlying_array)
    duplicate = ak.operations.copy(wrapper)
    underlying_array[2] = 123

    assert to_list(underlying_array) == [1.1, 2.2, 123.0, 4.4, 5.5]
    assert to_list(wrapper) == [1.1, 2.2, 123, 4.4, 5.5]
    assert to_list(duplicate) == [1.1, 2.2, 3.3, 4.4, 5.5]

    original = ak.Array([{"x": 1}, {"x": 2}, {"x": 3}])
    shallow_copy = copy.copy(original)
    shallow_copy["y"] = original.x**2
    assert to_list(shallow_copy) == [
        {"x": 1, "y": 1},
        {"x": 2, "y": 4},
        {"x": 3, "y": 9},
    ]
    assert to_list(original) == [{"x": 1}, {"x": 2}, {"x": 3}]

    array = ak.Array(
        [
            [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
            [],
            [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}],
        ]
    )

    assert to_list(ak.operations.copy(array)) == [
        [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
        [],
        [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}],
    ]

    a = ak.Array(
        [
            {"x": 0, "y": 0.0},
            {"x": 1, "y": 1.1},
            {"x": 2, "y": 2.2},
            {"x": 3, "y": 3.3},
            {"x": 4, "y": 4.4},
        ]
    )
    record = copy.deepcopy(a[2].layout)
    a["z"] = a.x**2

    assert to_list(a) == [
        {"x": 0, "y": 0.0, "z": 0},
        {"x": 1, "y": 1.1, "z": 1},
        {"x": 2, "y": 2.2, "z": 4},
        {"x": 3, "y": 3.3, "z": 9},
        {"x": 4, "y": 4.4, "z": 16},
    ]
    assert to_list(record) == {"x": 2, "y": 2.2}
    assert to_list(a[2]) == {"x": 2, "y": 2.2, "z": 4}
