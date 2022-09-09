# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test_one_level():
    base = ak._v2.operations.zip(
        {"a": ak._v2.operations.zip({"x": [1, 2, 3]}), "b": [1, 2, 3]},
        depth_limit=1,
    )
    what = ak._v2.Array([1.1, 2.2, 3.3], check_valid=True)
    assert to_list(ak._v2.operations.with_field(base, what, where=["a", "y"])) == [
        {"b": 1, "a": {"x": 1, "y": 1.1}},
        {"b": 2, "a": {"x": 2, "y": 2.2}},
        {"b": 3, "a": {"x": 3, "y": 3.3}},
    ]
    assert ak._v2.operations.fields(base) == ["a", "b"]

    base["a", "y"] = what
    assert to_list(base) == [
        {"b": 1, "a": {"x": 1, "y": 1.1}},
        {"b": 2, "a": {"x": 2, "y": 2.2}},
        {"b": 3, "a": {"x": 3, "y": 3.3}},
    ]


def test_two_level():
    base = ak._v2.operations.zip(
        {
            "A": ak._v2.operations.zip(
                {"a": ak._v2.operations.zip({"x": [1, 2, 3]}), "b": [1, 2, 3]}
            ),
            "B": [1, 2, 3],
        },
        depth_limit=1,
    )
    what = ak._v2.Array([1.1, 2.2, 3.3], check_valid=True)
    assert to_list(ak._v2.operations.with_field(base, what, where=["A", "a", "y"])) == [
        {"B": 1, "A": {"b": 1, "a": {"x": 1, "y": 1.1}}},
        {"B": 2, "A": {"b": 2, "a": {"x": 2, "y": 2.2}}},
        {"B": 3, "A": {"b": 3, "a": {"x": 3, "y": 3.3}}},
    ]
    assert ak._v2.operations.fields(base) == ["A", "B"]

    base["A", "a", "y"] = what
    assert to_list(base) == [
        {"B": 1, "A": {"b": 1, "a": {"x": 1, "y": 1.1}}},
        {"B": 2, "A": {"b": 2, "a": {"x": 2, "y": 2.2}}},
        {"B": 3, "A": {"b": 3, "a": {"x": 3, "y": 3.3}}},
    ]


def test_replace_the_only_field():
    base = ak._v2.operations.zip(
        {"a": ak._v2.operations.zip({"x": [1, 2, 3]})}, depth_limit=1
    )
    what = ak._v2.Array([1.1, 2.2, 3.3], check_valid=True)
    assert to_list(ak._v2.operations.with_field(base, what, where=["a", "y"])) == [
        {"a": {"x": 1, "y": 1.1}},
        {"a": {"x": 2, "y": 2.2}},
        {"a": {"x": 3, "y": 3.3}},
    ]
    assert ak._v2.operations.fields(base) == ["a"]

    base["a", "y"] = what
    assert to_list(base) == [
        {"a": {"x": 1, "y": 1.1}},
        {"a": {"x": 2, "y": 2.2}},
        {"a": {"x": 3, "y": 3.3}},
    ]


def test_check_no_field():
    base = ak._v2.operations.zip(
        {"a": ak._v2.operations.zip({"x": [1, 2, 3]})}, depth_limit=1
    )
    what = ak._v2.Array([1.1, 2.2, 3.3], check_valid=True)

    assert to_list(ak._v2.operations.with_field(base, what)) == [
        {"a": {"x": 1}, "1": 1.1},
        {"a": {"x": 2}, "1": 2.2},
        {"a": {"x": 3}, "1": 3.3},
    ]

    with pytest.raises(ValueError):
        ak._v2.operations.with_field(what, what)

    content1 = ak._v2.contents.NumpyArray(np.array([1, 2, 3]))
    recordarray = ak._v2.contents.RecordArray([content1], None)
    what = ak._v2.Array([1.1, 2.2, 3.3], check_valid=True)

    assert to_list(ak._v2.operations.with_field(recordarray, what)) == [
        (1, 1.1),
        (2, 2.2),
        (3, 3.3),
    ]
    assert to_list(ak._v2.operations.with_field(recordarray, what, "a")) == [
        {"0": 1, "a": 1.1},
        {"0": 2, "a": 2.2},
        {"0": 3, "a": 3.3},
    ]
    assert ak._v2.operations.fields(recordarray) == ["0"]
