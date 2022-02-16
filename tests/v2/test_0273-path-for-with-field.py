# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.convert.to_list


def test_one_level():

    base = ak.zip({"a": ak.zip({"x": [1, 2, 3]}), "b": [1, 2, 3]}, depth_limit=1)
    what = ak.Array([1.1, 2.2, 3.3], check_valid=True)
    assert ak.to_list(ak.with_field(base, what, where=["a", "y"])) == [
        {"b": 1, "a": {"x": 1, "y": 1.1}},
        {"b": 2, "a": {"x": 2, "y": 2.2}},
        {"b": 3, "a": {"x": 3, "y": 3.3}},
    ]
    base = ak._v2.operations.structure.zip(
        {"a": ak._v2.operations.structure.zip({"x": [1, 2, 3]}), "b": [1, 2, 3]},
        depth_limit=1,
    )
    what = ak._v2.Array([1.1, 2.2, 3.3], check_valid=True)
    assert to_list(
        ak._v2.operations.structure.with_field(base, what, where=["a", "y"])
    ) == [
        {"b": 1, "a": {"x": 1, "y": 1.1}},
        {"b": 2, "a": {"x": 2, "y": 2.2}},
        {"b": 3, "a": {"x": 3, "y": 3.3}},
    ]

    base["a", "y"] = what
    assert to_list(base) == [
        {"b": 1, "a": {"x": 1, "y": 1.1}},
        {"b": 2, "a": {"x": 2, "y": 2.2}},
        {"b": 3, "a": {"x": 3, "y": 3.3}},
    ]


def test_two_level():
    base = ak._v2.operations.structure.zip(
        {
            "A": ak._v2.operations.structure.zip(
                {"a": ak._v2.operations.structure.zip({"x": [1, 2, 3]}), "b": [1, 2, 3]}
            ),
            "B": [1, 2, 3],
        },
        depth_limit=1,
    )
    what = ak._v2.Array([1.1, 2.2, 3.3], check_valid=True)
    assert to_list(
        ak._v2.operations.structure.with_field(base, what, where=["A", "a", "y"])
    ) == [
        {"B": 1, "A": {"b": 1, "a": {"x": 1, "y": 1.1}}},
        {"B": 2, "A": {"b": 2, "a": {"x": 2, "y": 2.2}}},
        {"B": 3, "A": {"b": 3, "a": {"x": 3, "y": 3.3}}},
    ]

    base["A", "a", "y"] = what
    assert to_list(base) == [
        {"B": 1, "A": {"b": 1, "a": {"x": 1, "y": 1.1}}},
        {"B": 2, "A": {"b": 2, "a": {"x": 2, "y": 2.2}}},
        {"B": 3, "A": {"b": 3, "a": {"x": 3, "y": 3.3}}},
    ]


def test_replace_the_only_field():
    base = ak._v2.operations.structure.zip(
        {"a": ak._v2.operations.structure.zip({"x": [1, 2, 3]})}, depth_limit=1
    )
    what = ak._v2.Array([1.1, 2.2, 3.3], check_valid=True)
    assert to_list(
        ak._v2.operations.structure.with_field(base, what, where=["a", "y"])
    ) == [
        {"a": {"x": 1, "y": 1.1}},
        {"a": {"x": 2, "y": 2.2}},
        {"a": {"x": 3, "y": 3.3}},
    ]

    base["a", "y"] = what
    assert to_list(base) == [
        {"a": {"x": 1, "y": 1.1}},
        {"a": {"x": 2, "y": 2.2}},
        {"a": {"x": 3, "y": 3.3}},
    ]
