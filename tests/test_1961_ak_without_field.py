# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
import pytest

import awkward as ak


def test_one_level_record():
    base = ak.zip(
        {"x": [1, 2, 3], "y": [8, 9, 10]},
    )

    assert ak.without_field(base, where=["x"]).to_list() == [
        {"y": 8},
        {"y": 9},
        {"y": 10},
    ]
    assert ak.fields(base) == ["x", "y"]

    del base["x"]
    assert base.to_list() == [
        {"y": 8},
        {"y": 9},
        {"y": 10},
    ]


def test_one_level_tuple():
    base = ak.zip(
        [[1, 2, 3], [8, 9, 10]],
    )

    assert ak.without_field(base, where=["0"]).to_list() == [
        (8,),
        (9,),
        (10,),
    ]
    assert ak.fields(base) == ["0", "1"]

    del base["0"]
    assert base.to_list() == [
        (8,),
        (9,),
        (10,),
    ]


def test_two_level_record():
    base = ak.zip(
        {"a": ak.zip({"x": [1, 2, 3], "y": [8, 9, 10]}), "b": [1, 2, 3]},
        depth_limit=1,
    )

    assert ak.without_field(base, where=["a", "x"]).to_list() == [
        {"b": 1, "a": {"y": 8}},
        {"b": 2, "a": {"y": 9}},
        {"b": 3, "a": {"y": 10}},
    ]
    assert ak.fields(base) == ["a", "b"]
    assert ak.fields(base["a"]) == ["x", "y"]

    del base["a", "x"]
    assert base.to_list() == [
        {"b": 1, "a": {"y": 8}},
        {"b": 2, "a": {"y": 9}},
        {"b": 3, "a": {"y": 10}},
    ]


def test_two_level_tuple():
    base = ak.zip(
        [ak.zip([[1, 2, 3], [8, 9, 10]]), [1, 2, 3]],
        depth_limit=1,
    )

    assert ak.without_field(base, where=["0", "0"]).to_list() == [
        ((8,), 1),
        ((9,), 2),
        ((10,), 3),
    ]
    assert ak.fields(base) == ["0", "1"]
    assert ak.fields(base["0"]) == ["0", "1"]

    del base["0", "0"]
    assert base.to_list() == [
        ((8,), 1),
        ((9,), 2),
        ((10,), 3),
    ]


def test_two_level_mixed():
    base = ak.zip(
        [ak.zip({"x": [1, 2, 3], "y": [8, 9, 10]}), [1, 2, 3]],
        depth_limit=1,
    )

    assert ak.without_field(base, where=["0", "x"]).to_list() == [
        ({"y": 8}, 1),
        ({"y": 9}, 2),
        ({"y": 10}, 3),
    ]
    assert ak.fields(base) == ["0", "1"]
    assert ak.fields(base["0"]) == ["x", "y"]

    del base["0", "x"]
    assert base.to_list() == [
        ({"y": 8}, 1),
        ({"y": 9}, 2),
        ({"y": 10}, 3),
    ]


def test_one_level_delete_the_only_field():
    base = ak.zip({"x": [1, 2, 3]})
    assert ak.without_field(base, where=["x"]).to_list() == [
        {},
        {},
        {},
    ]
    assert ak.fields(base) == ["x"]

    del base["x"]
    assert base.to_list() == [
        {},
        {},
        {},
    ]


def test_two_level_delete_the_only_field():
    base = ak.zip({"a": ak.zip({"x": [1, 2, 3]})}, depth_limit=1)
    assert ak.without_field(base, where=["a", "x"]).to_list() == [
        {"a": {}},
        {"a": {}},
        {"a": {}},
    ]
    assert ak.fields(base) == ["a"]

    del base["a", "x"]
    assert base.to_list() == [
        {"a": {}},
        {"a": {}},
        {"a": {}},
    ]


def test_check_no_fields():
    base = ak.zip({"a": ak.zip({"x": [1, 2, 3]})}, depth_limit=1)

    with pytest.raises(IndexError, match=r"no field"):
        ak.without_field(base, "x")
