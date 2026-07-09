# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Regressions from the adversarial review of the lazy IR (CPU-runnable).

Every test pins a case where the default ``compute(fuse=True)`` path either
crashed or silently diverged from the eager result: non-finite folded
constants, deep loop-built chains (recursion limit / parser nesting limit),
string-array semantics, RegularArray and parameter preservation, lazy
``__getitem__`` masks, missing reflected operators, unbounded codegen caches,
and silent garbage from ``transform_lists`` on ragged input.
"""

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


@pytest.fixture
def arr():
    return ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0], [6.0, 7.0, 8.0, 9.0]])


# ----------------------------------------------------------------------
# Codegen crashes that must fall back to eager instead
# ----------------------------------------------------------------------


def test_inf_constant_matches_eager(arr):
    # repr(float("inf")) is "inf", which the generated source must resolve.
    expr = ak.cpu.lazy(arr) * float("inf")
    assert ak.to_list(expr.compute(fuse=True)) == ak.to_list(arr * float("inf"))


def test_nan_constant_matches_eager(arr):
    got = (ak.cpu.lazy(arr) + float("nan")).compute(fuse=True)
    assert all(np.isnan(x) for row in ak.to_list(got) for x in row)


def test_negative_inf_constant_matches_eager(arr):
    expr = ak.cpu.lazy(arr) + float("-inf")
    assert ak.to_list(expr.compute(fuse=True)) == ak.to_list(arr + float("-inf"))


@pytest.mark.parametrize("fuse", [True, False])
def test_deep_chain_does_not_hit_recursion_or_parser_limits(arr, fuse):
    # Loop-built expressions reach thousands of nodes: the fusion pass, the
    # eager-fallback evaluation, and the interpreter must all be iterative,
    # and a fused source too deep for the CPython parser (~200 nesting) must
    # fall back, not raise SyntaxError.
    n = 2500
    expr = ak.cpu.lazy(arr)
    for _ in range(n):
        expr = expr + 1
    assert ak.to_list(expr.compute(fuse=fuse)) == ak.to_list(arr + n)


def test_medium_chain_falls_back_cleanly(arr):
    # ~300 ops: fusion succeeds but its generated source exceeds the parser
    # nesting limit -> the executor must take the eager fallback silently.
    n = 300
    expr = ak.cpu.lazy(arr)
    for _ in range(n):
        expr = expr + 1
    result = expr.compute(fuse=True)
    assert ak.to_list(result) == ak.to_list(arr + n)
    assert expr.executor.fused_hits["eager"] >= 1


# ----------------------------------------------------------------------
# Fused results must not diverge from eager semantics
# ----------------------------------------------------------------------


def test_flat_string_equality_matches_eager():
    # A flat string array is a ListOffsetArray of chars; the fused flat pass
    # must not compare per-character where eager compares per-string.
    s = ak.Array(["ab", "cd"])
    s2 = ak.Array(["ab", "ce"])
    expr = ak.cpu.lazy(s) == ak.cpu.lazy(s2)
    assert ak.to_list(expr.compute(fuse=True)) == ak.to_list(s == s2) == [True, False]
    assert expr.executor.fused_hits["cpu"] == 0  # declined, not silently wrong


def test_nested_string_equality_matches_eager():
    s = ak.Array([["hello", "world"], ["hi"]])
    expr = ak.cpu.lazy(s) == ak.cpu.lazy(s)
    assert ak.to_list(expr.compute(fuse=True)) == ak.to_list(s == s)


def test_regular_array_type_preserved():
    r = ak.to_regular(ak.Array([[1.0, 2.0], [3.0, 4.0]]), axis=1)
    expr = ak.cpu.lazy(r) * 2
    fused = expr.compute(fuse=True)
    eager = r * 2
    assert str(fused.type) == str(eager.type)  # "2 * 2 * float64", not var
    assert ak.to_list(fused) == ak.to_list(eager)


def test_list_parameters_preserved():
    layout = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2, 3])),
        ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0])),
        parameters={"custom": "yes"},
    )
    parameterized = ak.Array(layout)
    fused = (ak.cpu.lazy(parameterized) * 2).compute(fuse=True)
    eager = parameterized * 2
    assert fused.layout.parameters == eager.layout.parameters
    assert ak.to_list(fused) == ak.to_list(eager)


def test_listarray_layout_matches_eager(arr):
    carried = arr[[2, 0, 1]]  # ListArray (starts/stops)
    fused = (ak.cpu.lazy(carried) * 2).compute(fuse=True)
    assert ak.to_list(fused) == ak.to_list(carried * 2)


# ----------------------------------------------------------------------
# __getitem__ with array keys (the arr[arr > 5] idiom)
# ----------------------------------------------------------------------


def test_getitem_with_lazy_boolean_mask(arr):
    la = ak.cpu.lazy(arr)
    got = la[la > 5.0].compute()
    assert ak.to_list(got) == ak.to_list(arr[arr > 5.0])


def test_getitem_with_lazy_mask_on_shared_subexpression(arr):
    la = ak.cpu.lazy(arr)
    t = la * 2 + 1
    got = t[t > 5.0].compute(fuse=True)
    eager = (arr * 2 + 1)[(arr * 2 + 1) > 5.0]
    assert ak.to_list(got) == ak.to_list(eager)


def test_getitem_with_eager_array_key(arr):
    mask = arr > 5.0
    got = ak.cpu.lazy(arr)[mask].compute()
    assert ak.to_list(got) == ak.to_list(arr[mask])


def test_getitem_plain_keys_still_work():
    rec = ak.Array([[{"x": 1.0}], [{"x": 2.0}, {"x": 3.0}]])
    la = ak.cpu.lazy(rec)
    assert ak.to_list(la["x"].compute()) == ak.to_list(rec["x"])
    assert ak.to_list(la[1:].compute()) == ak.to_list(rec[1:])


# ----------------------------------------------------------------------
# Operator coverage: reflected and additional binary ops
# ----------------------------------------------------------------------


def test_rpow_matches_eager(arr):
    assert ak.to_list((2 ** ak.cpu.lazy(arr)).compute()) == ak.to_list(2**arr)


def test_mod_and_floordiv_match_eager(arr):
    la = ak.cpu.lazy(arr)
    assert ak.to_list((la % 2).compute()) == ak.to_list(arr % 2)
    assert ak.to_list((la // 2).compute()) == ak.to_list(arr // 2)
    assert ak.to_list((7 % la).compute()) == ak.to_list(7 % arr)
    assert ak.to_list((7 // la).compute()) == ak.to_list(7 // arr)


def test_neg_matches_eager(arr):
    fused = (-ak.cpu.lazy(arr)).compute(fuse=True)
    eager = -arr
    assert ak.to_list(fused) == ak.to_list(eager)
    assert fused.layout.content.dtype == eager.layout.content.dtype


def test_mod_fuses_into_region(arr):
    la = ak.cpu.lazy(arr)
    expr = (la * 2) % 3
    assert ak.to_list(expr.compute(fuse=True)) == ak.to_list((arr * 2) % 3)
    assert expr.executor.fused_hits["cpu"] == 1


# ----------------------------------------------------------------------
# Codegen cache must be bounded (constants are folded into the source)
# ----------------------------------------------------------------------


def test_compile_op_caches_are_bounded():
    from awkward._connect.cpu._fusion_codegen import _compile_op as cpu_op
    from awkward._connect.cuda._fusion_codegen import _compile_op as cuda_op

    assert cpu_op.cache_info().maxsize is not None
    assert cuda_op.cache_info().maxsize is not None


# ----------------------------------------------------------------------
# transform_lists: precondition enforced, output buffer returned
# ----------------------------------------------------------------------


def test_transform_lists_ragged_raises():
    from awkward._connect.cpu.helpers import transform_lists

    ragged = ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0], [6.0]])
    with pytest.raises(ValueError, match="exactly 2"):
        transform_lists(ragged, np.zeros(3), 2, lambda x, y: x + y)


def test_transform_lists_returns_written_buffer():
    from awkward._connect.cpu.helpers import transform_lists

    pairs = ak.Array([[1.0, 2.0], [3.0, 4.0]])
    out = transform_lists(pairs, np.zeros(2), 2, lambda x, y: x + y)
    assert out.tolist() == [3.0, 7.0]
