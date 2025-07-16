# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
from __future__ import annotations

import awkward as ak


def test_to_lists_of_records():
    a = ak.Array(
        [
            {"a": [1, 2, 3], "b": [4, 5, 6]},
            {"a": [7, 8], "b": [9, 10]},
            {"a": [], "b": []},
        ]
    )
    assert ak.to_lists_of_records(a).typestr == "3 * var * {a: int64, b: int64}"
    assert ak.to_lists_of_records(a).tolist() == [
        [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}],
        [{"a": 7, "b": 9}, {"a": 8, "b": 10}],
        [],
    ]


def test_to_lists_of_records_tuple():
    t = ak.Array([([1, 2, 3], [4, 5, 6]), ([7, 8], [9, 10]), ([], [])])
    assert ak.to_lists_of_records(t).typestr == "3 * var * (int64, int64)"
    assert ak.to_lists_of_records(t).tolist() == [
        [(1, 4), (2, 5), (3, 6)],
        [(7, 9), (8, 10)],
        [],
    ]


def test_to_lists_of_records_2D():
    b = ak.Array(
        [
            [{"a": [1, 2, 3], "b": [4, 5, 6]}],
            [{"a": [7, 8], "b": [9, 10]}, {"a": [], "b": []}],
        ]
    )
    assert ak.to_lists_of_records(b).typestr == "2 * var * var * {a: int64, b: int64}"
    assert ak.to_lists_of_records(b).tolist() == [
        [[{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]],
        [[{"a": 7, "b": 9}, {"a": 8, "b": 10}], []],
    ]


def test_to_lists_of_records_depth_limit_2():
    c = ak.Array(
        [{"a": [[1, 2, 3]], "b": [[4, 5, 6]]}, {"a": [[7, 8], []], "b": [[9, 10], []]}]
    )
    assert (
        ak.to_lists_of_records(c, depth_limit=2).typestr
        == "2 * var * {a: var * int64, b: var * int64}"
    )
    assert ak.to_lists_of_records(c, depth_limit=2).tolist() == [
        [{"a": [1, 2, 3], "b": [4, 5, 6]}],
        [{"a": [7, 8], "b": [9, 10]}, {"a": [], "b": []}],
    ]


def test_to_record_of_lists():
    a = ak.Array(
        [
            [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}],
            [{"a": 7, "b": 9}, {"a": 8, "b": 10}],
            [],
        ]
    )
    assert ak.to_record_of_lists(a).typestr == "3 * {a: var * int64, b: var * int64}"
    assert ak.to_record_of_lists(a).tolist() == [
        {"a": [1, 2, 3], "b": [4, 5, 6]},
        {"a": [7, 8], "b": [9, 10]},
        {"a": [], "b": []},
    ]


def test_to_record_of_lists_tuple():
    t = ak.Array(
        [
            [(1, 4), (2, 5), (3, 6)],
            [(7, 9), (8, 10)],
            [],
        ]
    )
    assert ak.to_record_of_lists(t).typestr == "3 * (var * int64, var * int64)"
    assert ak.to_record_of_lists(t).tolist() == [
        ([1, 2, 3], [4, 5, 6]),
        ([7, 8], [9, 10]),
        ([], []),
    ]


def test_to_record_of_lists_axis_1():
    b = ak.Array(
        [
            [[{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]],
            [[{"a": 7, "b": 9}, {"a": 8, "b": 10}], []],
        ]
    )
    assert (
        ak.to_record_of_lists(b, axis=1).typestr
        == "2 * var * {a: var * int64, b: var * int64}"
    )
    assert ak.to_record_of_lists(b, axis=1).tolist() == [
        [{"a": [1, 2, 3], "b": [4, 5, 6]}],
        [{"a": [7, 8], "b": [9, 10]}, {"a": [], "b": []}],
    ]


def test_to_record_of_lists_axis_1_named():
    b = ak.Array(
        [
            [[{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]],
            [[{"a": 7, "b": 9}, {"a": 8, "b": 10}], []],
        ],
        named_axis=("outer", "inner"),
    )
    assert (
        ak.to_record_of_lists(b, axis="inner").typestr
        == "2 * var * {a: var * int64, b: var * int64}"
    )
    assert ak.to_record_of_lists(b, axis="inner").tolist() == [
        [{"a": [1, 2, 3], "b": [4, 5, 6]}],
        [{"a": [7, 8], "b": [9, 10]}, {"a": [], "b": []}],
    ]
