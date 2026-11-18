# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import itertools

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
@pytest.mark.parametrize("optional", [False, True])
@pytest.mark.parametrize("mergebool", [False, True])
def test_simplification_reconsiders_branches_after_merging(order, optional, mergebool):
    values = [[1], ["a"], [2, "b"]]
    contents = [
        ak.to_layout([values[i], None] if optional else [values[i]]) for i in order
    ]
    tags = ak.index.Index8(np.array([0, 1, 2] * (2 if optional else 1), dtype=np.int8))
    index = ak.index.Index64(
        np.array([0, 0, 0] + ([1, 1, 1] if optional else []), dtype=np.int64)
    )
    kwargs = {"mergebool": mergebool, "parameters": {"source": "test"}}
    result = ak.contents.UnionArray.simplified(tags, index, contents, **kwargs)
    assert result.to_list() == [values[i] for i in order] + (
        [None] * 3 if optional else []
    )
    assert not result.is_union
    assert result.parameter("source") == "test"
    assert ak.validity_error(result) == ""
    traced = ak.contents.UnionArray.simplified(
        tags.to_nplike(ak._nplikes.typetracer.TypeTracer.instance()),
        index.to_nplike(ak._nplikes.typetracer.TypeTracer.instance()),
        [content.to_typetracer(forget_length=True) for content in contents],
        **kwargs,
    )
    assert traced.form == result.form


def test_irreducible_union_remains_unchanged():
    contents = [ak.to_layout([1, 2]), ak.to_layout(["a", "b"])]
    tags = ak.index.Index8(np.array([1, 0, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([1, 0, 0], dtype=np.int64))
    result = ak.contents.UnionArray.simplified(tags, index, contents)
    assert result.is_union
    assert result.to_list() == ["b", 1, "a"]
    again = ak.contents.UnionArray.simplified(
        result.tags, result.index, result.contents
    )
    assert result.is_equal_to(again)
