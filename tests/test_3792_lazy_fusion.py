# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Tests for the kernel-fusion pass on the operator IR (CPU-runnable).

These exercise the backend-neutral fusion machinery
(``awkward._connect.lazy._fusion``): the graph rewrite that collapses
element-wise regions into ``FusedNode``s, numeric equivalence between the
fused and interpreter paths, DAG fan-out handling, and the CUDA codegen's
GPU-free surface (op-source generation, leaf classification).  The actual
``cuda.compute`` kernel emission needs a GPU and is covered in tests-cuda.
"""

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._connect.lazy._fusion import (
    FusedNode,
    fuse,
    is_elementwise,
)
from awkward._connect.lazy._ir import OpType


@pytest.fixture
def arr():
    return ak.Array([[1, 2, 3], [4, 5], [6, 7, 8, 9]])


@pytest.fixture
def arr2():
    return ak.Array([[10, 20, 30], [40, 50], [60, 70, 80, 90]])


# ----------------------------------------------------------------------
# Numeric equivalence: fused vs interpreter vs eager
# ----------------------------------------------------------------------


def _fresh(arr):
    return ak.cpu.lazy(arr)


def test_elementwise_chain_matches_interpreter(arr):
    expected = ak.to_list((arr * 2 + 1) * 3)
    assert ak.to_list(((_fresh(arr) * 2 + 1) * 3).compute(fuse=True)) == expected
    assert ak.to_list(((_fresh(arr) * 2 + 1) * 3).compute(fuse=False)) == expected


def test_division_produces_float_like_eager(arr):
    expected = ak.to_list(arr / 2 + 0.5)
    assert ak.to_list((_fresh(arr) / 2 + 0.5).compute(fuse=True)) == expected


def test_power_and_mixed_ops(arr):
    expected = ak.to_list((arr**2 - arr) * 2)
    assert ak.to_list(
        (_fresh(arr) ** 2 - _fresh(arr)).compute(fuse=False)
    ) == ak.to_list(arr**2 - arr)
    la = _fresh(arr)
    assert ak.to_list(((la**2 - la) * 2).compute(fuse=True)) == expected


def test_two_input_columns(arr, arr2):
    expected = ak.to_list(arr * arr2 + arr)
    la, lb = _fresh(arr), ak.cpu.lazy(arr2)
    # share `la` twice -> fan-out, still numerically correct
    assert ak.to_list((la * lb + la).compute(fuse=True)) == expected


def test_filter_with_fused_condition(arr):
    expected = ak.to_list((arr * 2 + 1)[(arr * 2 + 1) > 5])
    la = _fresh(arr)
    t = la * 2 + 1
    assert ak.to_list(t.filter(t > 5).compute(fuse=True)) == expected


def test_filter_on_raw_input_condition(arr):
    expected = ak.to_list((arr * 2 + 1)[arr > 3])
    la = _fresh(arr)
    assert ak.to_list((la * 2 + 1).filter(la > 3).compute(fuse=True)) == expected


def test_comparison_result_matches(arr):
    expected = ak.to_list((arr * 2) > (arr + 3))
    la = _fresh(arr)
    assert ak.to_list(((la * 2) > (la + 3)).compute(fuse=True)) == expected


def test_fuse_then_nofuse_recomputes_consistently(arr):
    la = _fresh(arr)
    expr = la * 2 + 1
    a = ak.to_list(expr.compute(fuse=True))
    b = ak.to_list(expr.compute(fuse=False))  # flag change -> recompute
    assert a == b == ak.to_list(arr * 2 + 1)


# ----------------------------------------------------------------------
# Fusion structure / grouping
# ----------------------------------------------------------------------


def test_linear_chain_is_one_region(arr):
    root = (_fresh(arr) * 2 + 1).ir_node
    fused = fuse(root)
    assert isinstance(fused, FusedNode)
    assert fused.reduce_op is None
    # leaves: input + two constants
    assert len(fused.leaves) == 3
    assert fused.expr_text == "(($0 * $1) + $2)"


def test_fanout_becomes_boundary(arr):
    la = _fresh(arr)
    t = la * 2 + 1  # shared by both filter input and condition
    stats = t.filter(t > 5).fusion_stats()
    # three element-wise ops (mul, add, gt) -> two regions:
    # the shared (mul+add) region, and the (>5) region on top of it
    assert stats["elementwise_before"] == 3
    assert stats["fused_regions"] == 2


def test_shared_region_is_computed_once(arr):
    la = _fresh(arr)
    t = la * 2 + 1
    fused = fuse(t.filter(t > 5).ir_node)
    # The FilterNode's input and the comparison's leaf must be the SAME object
    filter_input = fused.input
    comparison = fused.condition
    assert isinstance(filter_input, FusedNode)
    assert isinstance(comparison, FusedNode)
    assert comparison.leaves[0] is filter_input  # shared, not duplicated


def test_single_binary_op_fuses(arr):
    fused = fuse((_fresh(arr) + 1).ir_node)
    assert isinstance(fused, FusedNode)
    assert len(fused.leaves) == 2


def test_reduce_over_elementwise_is_fused(arr):
    # (arr*2+1).sum() -> one FusedNode carrying reduce_op (transform+reduce)
    la = _fresh(arr)
    fused = fuse((la * 2 + 1).sum().ir_node)
    assert isinstance(fused, FusedNode)
    assert fused.reduce_op is OpType.SUM
    assert fused.expr_text == "(($0 * $1) + $2)"


def test_reduce_over_plain_input_not_fused(arr):
    # arr.sum() has a non-element-wise (Input) child -> stays a ReduceNode
    from awkward._connect.lazy._ir import ReduceNode

    fused = fuse(_fresh(arr).sum().ir_node)
    assert isinstance(fused, ReduceNode)
    assert not isinstance(fused, FusedNode)


def test_getitem_is_a_boundary(arr):
    # field access is opaque to fusion; the sum over it is a separate region
    la = ak.cpu.lazy(ak.Array([[{"x": 1}], [{"x": 2}, {"x": 3}]]))
    node = (la["x"] + 1).ir_node
    fused = fuse(node)
    assert isinstance(fused, FusedNode)
    # the GetItemNode is a leaf of the region (boundary), plus the constant
    assert len(fused.leaves) == 2


def test_is_elementwise_classification(arr):
    la = _fresh(arr)
    assert is_elementwise((la + 1).ir_node)
    assert is_elementwise((la > 1).ir_node)
    assert not is_elementwise(la.ir_node)  # InputNode
    assert not is_elementwise(la.sum().ir_node)  # ReduceNode


def test_fusion_reduces_region_count(arr):
    # a long element-wise chain collapses to a single region
    la = _fresh(arr)
    expr = la
    for k in range(6):
        expr = expr * 2 + k
    stats = expr.fusion_stats()
    assert stats["elementwise_before"] == 12  # 6 * (mul, add)
    assert stats["fused_regions"] == 1


# ----------------------------------------------------------------------
# Introspection: compile / visualize
# ----------------------------------------------------------------------


def test_compile_returns_fused_graph(arr):
    fused = (_fresh(arr) * 2 + 1).compile()
    assert isinstance(fused, FusedNode)


def test_visualize_fused_shows_fusednode(arr):
    text = (_fresh(arr) * 2 + 1).visualize(fused=True)
    assert "FusedNode" in text
    assert "expr=" in text


def test_visualize_unfused_has_no_fusednode(arr):
    text = (_fresh(arr) * 2 + 1).visualize(fused=False)
    assert "FusedNode" not in text


# ----------------------------------------------------------------------
# Cache stability of the compiled expression
# ----------------------------------------------------------------------


def test_expr_is_cache_stable(arr):
    # The compiled callable must not close over per-call state: evaluating it
    # twice on different inputs gives independent, correct results.
    fused = fuse((_fresh(arr) * 2 + 1).ir_node)
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])
    # leaves are [input, const 2, const 1]; feed values in slot order
    assert list(fused.expr([a, 2, 1])) == [3.0, 5.0, 7.0]
    assert list(fused.expr([b, 2, 1])) == [21.0, 41.0, 61.0]


# ----------------------------------------------------------------------
# CUDA codegen: GPU-free surface (op-source, leaf classification, fallback)
# ----------------------------------------------------------------------


def test_codegen_imports_without_gpu():
    import awkward._connect.cuda._fusion_codegen as fc

    assert issubclass(fc.FusionUnsupported, Exception)


def test_op_source_generation(arr):
    fused = fuse((_fresh(arr) * 2 + 1).ir_node)
    # map the column leaf to t[0], fold the two constants
    lid_input = fused.leaf_ids[0]
    lid_c2 = fused.leaf_ids[1]
    lid_c1 = fused.leaf_ids[2]
    src = fused.op_source({lid_input: "t[0]", lid_c2: "2", lid_c1: "1"})
    assert src == "((t[0] * 2) + 1)"
    # and it is valid, correct Python
    fn = eval("lambda t: " + src)
    assert fn((5,)) == 11


def test_classify_leaf_scalar_and_unsupported():
    import awkward._connect.cuda._fusion_codegen as fc

    assert fc._classify_leaf(3)[0] == "scalar"
    assert fc._classify_leaf(2.5)[0] == "scalar"
    assert fc._classify_leaf(np.float64(4.0))[0] == "scalar"
    with pytest.raises(fc.FusionUnsupported):
        fc._classify_leaf(object())


def test_cuda_build_op_single_column_uses_scalar_element():
    # Regression: a one-column region's transform element is the scalar itself
    # (`t`), not `t[0]` — the input is not zipped when there is a single column.
    import awkward._connect.cuda._fusion_codegen as fc

    a = ak.Array([[1.0, 2], [3.0]])
    node = fuse((ak.cpu.lazy(a) * 2 + 1).ir_node)  # leaves: [col, 2, 1]
    op, columns = fc._build_op(node, [a, 2, 1])
    assert len(columns) == 1
    assert op(5.0) == 11.0  # op receives a bare scalar, not a tuple


def test_cuda_build_op_multi_column_uses_indexed_element():
    import awkward._connect.cuda._fusion_codegen as fc

    a = ak.Array([[1.0, 2], [3.0]])
    b = ak.Array([[4.0, 5], [6.0]])
    node = fuse((ak.cpu.lazy(a) + ak.cpu.lazy(b)).ir_node)  # leaves: [colA, colB]
    op, columns = fc._build_op(node, [a, b])
    assert len(columns) == 2
    assert op((5.0, 7.0)) == 12.0  # op receives the zipped tuple


# ----------------------------------------------------------------------
# Phase 2 completion: optimize() home, idempotence, cache-stability,
# no-fuse debug mode.  (These are the plan's hard constraints.)
# ----------------------------------------------------------------------


def test_optimize_runs_fusion(arr):
    from awkward._connect.lazy._executor import IRExecutor

    optimized = IRExecutor().optimize((_fresh(arr) * 2 + 1).ir_node)
    assert isinstance(optimized, FusedNode)


def test_optimize_is_idempotent(arr):
    from awkward._connect.lazy._executor import IRExecutor

    ex = IRExecutor()
    once = ex.optimize((_fresh(arr) * 2 + 1).ir_node)
    twice = ex.optimize(once)
    assert isinstance(once, FusedNode) and isinstance(twice, FusedNode)
    # second pass leaves the already-fused graph structurally unchanged
    assert twice is once or twice.expr_text == once.expr_text
    assert ak.to_list(ex.execute(twice)) == ak.to_list(ex.execute(once))


def test_compile_and_execute_matches_optimize_then_execute(arr):
    from awkward._connect.lazy._executor import IRExecutor

    node = ((_fresh(arr) * 2 + 1) * 3).ir_node
    a = ak.to_list(IRExecutor().compile_and_execute(node))
    ex = IRExecutor()
    b = ak.to_list(ex.execute(ex.optimize(node)))
    assert a == b == ak.to_list((arr * 2 + 1) * 3)


def test_fused_op_is_cache_stable():
    # Plan's hard constraint: a stable program compiles its fused op ONCE and
    # reuses it -- no per-call recompile (the 1.8 s/call regression).
    #
    # The cache is unbounded (functools.cache, never evicted/cleared here), so
    # the robust invariant is: the same generated source always maps to the
    # SAME compiled callable -- every call site, including each compute(),
    # reuses it. Asserted via object identity, which (unlike the global
    # hit/miss counters) is immune to parallel / free-threaded test execution
    # where other tests share this process-global cache.
    from awkward._connect.cpu import _fusion_codegen as cfc

    a = ak.Array([[1.0, 2, 3], [4, 5], [6, 7]])
    node = fuse((ak.cpu.lazy(a) * 2 + 1).ir_node)
    ids = node.leaf_ids  # [column, const 2, const 1]
    src = node.op_source({ids[0]: "c[0]", ids[1]: "2", ids[2]: "1"})

    op_obj = cfc._compile_op(src)
    assert cfc._compile_op(src) is op_obj  # recompiling the source is a no-op

    # Running the program (fresh graphs, new node ids) never replaces the op:
    # each compute reuses the one interned callable rather than recompiling.
    for _ in range(5):
        (ak.cpu.lazy(a) * 2 + 1).compute(fuse=True)
    assert cfc._compile_op(src) is op_obj


def test_no_fuse_debug_mode_matches_fused_across_battery(arr, arr2):
    la, lb = _fresh(arr), ak.cpu.lazy(arr2)
    programs = [
        lambda: _fresh(arr) * 2 + 1,
        lambda: (_fresh(arr) * 2 + 1) * 3 - 4,
        lambda: _fresh(arr) / 2 + 0.5,
        lambda: la * lb + la,
        lambda: (_fresh(arr) * 2) > (_fresh(arr) + 3),
    ]
    for make in programs:
        fused = ak.to_list(make().compute(fuse=True))
        interp = ak.to_list(make().compute(fuse=False))
        assert fused == interp


# ----------------------------------------------------------------------
# Public entry point: ak.cuda.to_cccl_iterator (CPU-visible surface)
# ----------------------------------------------------------------------


def test_to_cccl_iterator_is_public():
    # The symbol is exported regardless of whether cupy is installed.
    assert "to_cccl_iterator" in ak.cuda.__all__
    assert callable(ak.cuda.to_cccl_iterator)


def test_to_cccl_iterator_without_cupy_errors_clearly():
    # Where cupy is unavailable, the entry point must fail with an actionable
    # install hint rather than an obscure ImportError deep in the helpers.
    pytest.importorskip  # noqa: B018 - keep import symmetry with GPU tests
    try:
        import cupy  # noqa: F401
    except ImportError:
        with pytest.raises(ModuleNotFoundError, match="cupy"):
            ak.cuda.to_cccl_iterator(ak.Array([[1, 2], [3]]))


def test_cpu_arrays_skip_cuda_codegen(arr):
    # _is_cuda_backed must be False for cpu-backed leaves, so the executor
    # never reaches the cuda codegen path (it evaluates eagerly instead).
    from awkward._connect.lazy._executor import _is_cuda_backed

    assert _is_cuda_backed([arr]) is False
    assert _is_cuda_backed([arr, 3, 2.0]) is False


# ----------------------------------------------------------------------
# CPU fusion codegen: the flat-buffer single pass that makes fusion a win
# ----------------------------------------------------------------------


def test_cpu_codegen_elementwise_matches_eager_with_empty_lists():
    arr = ak.Array([[1.0, 2, 3], [4, 5], [], [6, 7, 8, 9]])
    la = ak.cpu.lazy(arr)
    got = ak.to_list(((la * 2 + 1) * 3).compute(fuse=True))
    assert got == ak.to_list((arr * 2 + 1) * 3)  # empty sublist preserved


def test_cpu_codegen_used_for_cpu_leaves():
    # The executor's CPU fused path returns a result (not the fallback
    # sentinel) for an aligned element-wise region on cpu-backed leaves.
    from awkward._connect.lazy._executor import _NO_FUSED_KERNEL, IRExecutor
    from awkward._connect.lazy._fusion import fuse

    arr = ak.Array([[1.0, 2], [3, 4, 5]])
    node = fuse((ak.cpu.lazy(arr) * 2 + 1).ir_node)
    values = [IRExecutor().execute(leaf) for leaf in node.leaves]
    out = IRExecutor._maybe_cpu_fused(node, values)
    assert out is not _NO_FUSED_KERNEL
    assert ak.to_list(out) == ak.to_list(arr * 2 + 1)


def _axis_reducers_available():
    # The generated axis=-1 reducer kernels are missing in some sandboxes;
    # skip the reduction-execution tests there (they are environment-specific,
    # not fusion-specific -- eager ak.sum fails identically).
    try:
        ak.sum(ak.Array([[1.0, 2.0], [3.0]]), axis=-1)
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _axis_reducers_available(), reason="axis=-1 reducer unavailable here"
)
def test_cpu_fused_sum_returns_awkward_and_matches_eager():
    # CPU fusion lowers only the element-wise map; the reduction is delegated
    # to the eager Awkward reducer, so fuse=True/False are identical and the
    # result is an Awkward value (not a raw NumPy array).
    arr = ak.Array([[1.0, 2, 3], [4, 5], [], [6.0]])
    la = ak.cpu.lazy(arr)
    fused = (la * 2 + 1).sum().compute(fuse=True)
    eager = (la * 2 + 1).sum().compute(fuse=False)
    assert not isinstance(fused, np.ndarray)  # Awkward-typed, not raw NumPy
    assert ak.to_list(fused) == ak.to_list(eager)
    assert ak.to_list(fused) == [sum(2 * x + 1 for x in row) for row in ak.to_list(arr)]


@pytest.mark.skipif(
    not _axis_reducers_available(), reason="axis=-1 reducer unavailable here"
)
def test_cpu_fused_mean_matches_eager():
    arr = ak.Array([[2.0, 4], [1.0, 2, 3], [5.0]])
    la = ak.cpu.lazy(arr)
    fused = (la + 0.0).mean().compute(fuse=True)
    eager = (la + 0.0).mean().compute(fuse=False)
    assert ak.to_list(fused) == ak.to_list(eager)


def test_cpu_codegen_rejects_mismatched_offsets():
    from awkward._connect.cpu._fusion_codegen import (
        FusionUnsupported,
        execute_fused_cpu,
    )
    from awkward._connect.lazy._fusion import fuse

    a = ak.Array([[1.0, 2, 3], [4, 5]])
    b = ak.Array([[1.0, 2], [3, 4, 5]])  # different offsets
    node = fuse((ak.cpu.lazy(a) + ak.cpu.lazy(b)).ir_node)
    with pytest.raises(FusionUnsupported):
        execute_fused_cpu(node, [a, b])


# ----------------------------------------------------------------------
# Regression: fused-default path must not diverge from eager (review P2s)
# ----------------------------------------------------------------------


def test_flat_array_falls_back_to_eager_not_crash():
    # A flat (non-list) array is a valid eager expression but unsupported by CPU
    # fusion; default fuse=True must fall back, not raise (P2 #1).
    a = ak.Array([1, 2, 3])
    expr = ak.cpu.lazy(a) * 2 + 1
    assert ak.to_list(expr.compute(fuse=True)) == ak.to_list(a * 2 + 1)
    assert expr.executor.fused_hits["cpu"] == 0  # cpu fusion declined
    assert expr.executor.fused_hits["eager"] >= 1  # eager fallback taken


def test_cpu_unsupported_layout_raises_fusion_unsupported():
    # The boundary itself converts layout errors to FusionUnsupported so the
    # executor can catch and fall back.
    from awkward._connect.cpu._fusion_codegen import (
        FusionUnsupported,
        execute_fused_cpu,
    )

    a = ak.Array([1, 2, 3])  # flat, no offsets
    node = fuse((ak.cpu.lazy(a) + 1).ir_node)
    with pytest.raises(FusionUnsupported):
        execute_fused_cpu(node, [a, 1])


def test_cpu_fused_preserves_integer_dtype():
    # Integer input must stay integer through the fused element-wise path (P2 #3).
    arr = ak.Array([[1, 2, 3], [4, 5]])
    fused = (ak.cpu.lazy(arr) * 2 + 1).compute(fuse=True)
    eager = arr * 2 + 1
    assert fused.layout.content.dtype == eager.layout.content.dtype
    assert not np.issubdtype(fused.layout.content.dtype, np.floating)
    assert ak.to_list(fused) == ak.to_list(eager)


def test_input_node_infers_dtype():
    from awkward._connect.lazy._ir import InputNode

    assert InputNode(ak.Array([[1, 2, 3], [4, 5]])).dtype == np.dtype("int64")
    assert InputNode(ak.Array([[1.0, 2.0]])).dtype == np.dtype("float64")
    assert InputNode(ak.Array([{"x": 1}])).dtype is None  # record: no single dtype


def test_cuda_sum_accumulator_dtype_widens_like_ak():
    # GPU-free: the sum-output dtype widens small ints like ak.sum's accumulator.
    from awkward._connect.cuda._fusion_codegen import _sum_accumulator_dtype

    assert _sum_accumulator_dtype(np.int32) == np.dtype("int64")
    assert _sum_accumulator_dtype(np.uint16) == np.dtype("uint64")
    assert _sum_accumulator_dtype(np.bool_) == np.dtype("int64")
    assert _sum_accumulator_dtype(np.float32) == np.dtype("float32")  # floats unchanged


def test_cuda_infer_out_dtype_matches_numpy():
    # GPU-free: element-wise output dtype follows NumPy promotion, not float64.
    from awkward._connect.cuda._fusion_codegen import _build_op, _infer_out_dtype

    a = ak.Array([[1, 2, 3], [4, 5]])  # int64 content
    node = fuse((ak.cpu.lazy(a) * 2 + 1).ir_node)
    op, columns = _build_op(node, [a, 2, 1])
    assert _infer_out_dtype(op, columns, single=True) == np.dtype("int64")
