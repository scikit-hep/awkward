# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    x = ak.with_parameter([1, 2, np.datetime64("now")], "x", 100).layout
    y = ak.contents.NumpyArray(np.array([99, 88, 77], dtype=np.int64))
    result = ak.contents.UnionArray.simplified(
        ak.index.Index8(np.array([0, 0, 1, 1], dtype=np.int8)),
        ak.index.Index64(np.array([0, 1, 0, 1], dtype=np.int64)),
        [x, y],
        parameters={"x": 100},
    )
    assert result.parameters == {"x": 100}

    result2 = ak.contents.UnionArray.simplified(
        ak.index.Index8(np.array([0, 0, 1, 1])),
        ak.index.Index64(np.array([0, 1, 0, 1])),
        [x, y],
        parameters={"x": 200},
    )
    assert result2.parameters == {"x": 200}

    result3 = ak.contents.UnionArray.simplified(
        ak.index.Index8(np.array([0, 0, 1, 1])),
        ak.index.Index64(np.array([0, 1, 0, 1])),
        [x, y],
    )
    assert result3.parameters == {"x": 100}


def test_merge_optional_strings():
    a = ak.Array([{"a": "foo"}, {"a": "bar"}])
    b = ak.Array([{"a": None}])

    result = ak.concatenate([a, b])
    assert ak.almost_equal(result, [{"a": "foo"}, {"a": "bar"}, {"a": None}])


def test_merge_permutations():
    string_optional = ak.Array([{"a": "foo"}, {"a": None}])[0:0]
    kinds = {
        "optional": string_optional,
        "empty": ak.Array([{"a": None}]),
        "string": ak.Array([{"a": "baz"}]),
    }
    for perm in permutations(kinds, 3):
        assert str(ak.concatenate([kinds[k] for k in perm]).type) == "2 * {a: ?string}"


def test_merge_type_commutativity():
    one = ak.concatenate((ak.with_parameter([1], "k", "v"), [None]))
    two = ak.concatenate(([None], ak.with_parameter([1], "k", "v")))
    assert type(one) == type(two)
