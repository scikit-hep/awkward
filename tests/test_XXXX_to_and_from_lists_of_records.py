# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import awkward as ak


def test_to_lists_of_records():
    a = ak.Array(
        [
            {"a": [1, 2, 3], "b": [4, 5, 6]},
            {"a": [7, 8], "b": [9, 10]},
            {"a": [], "b": []},
        ]
    )
    assert ak.to_lists_of_records(a).tolist() == [
        [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}],
        [{"a": 7, "b": 9}, {"a": 8, "b": 10}],
        [],
    ]


def test_to_lists_of_records_axis_1():
    b = ak.Array(
        [
            [{"a": [1, 2, 3], "b": [4, 5, 6]}],
            [{"a": [7, 8], "b": [9, 10]}, {"a": [], "b": []}],
        ]
    )
    assert ak.to_lists_of_records(b, axis=1).tolist() == [
        [[{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]],
        [[{"a": 7, "b": 9}, {"a": 8, "b": 10}], []],
    ]


def test_from_lists_of_records():
    a = ak.Array(
        [
            [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}],
            [{"a": 7, "b": 9}, {"a": 8, "b": 10}],
            [],
        ]
    )
    assert ak.from_lists_of_records(a).tolist() == [
        {"a": [1, 2, 3], "b": [4, 5, 6]},
        {"a": [7, 8], "b": [9, 10]},
        {"a": [], "b": []},
    ]


def test_from_lists_of_records_axis_1():
    b = [
        [[{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]],
        [[{"a": 7, "b": 9}, {"a": 8, "b": 10}], []],
    ]
    assert ak.from_lists_of_records(b, axis=1).tolist() == [
            [{"a": [1, 2, 3], "b": [4, 5, 6]}],
            [{"a": [7, 8], "b": [9, 10]}, {"a": [], "b": []}],
        ]
