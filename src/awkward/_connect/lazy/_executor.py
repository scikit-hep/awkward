# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Interpreter for the operator-level IR (backend-neutral).

Each node dispatches to an eager Awkward operation, which in turn dispatches
to the array's backend (NumPy kernels for cpu, CUDA kernels for cuda).
Predicate-based CCCL fast paths live in ``awkward._connect.cuda.helpers``
and ``awkward._connect.cuda.ir_nodes``; they require a callable predicate
rather than a materialized boolean array, so they are not used here.
"""

from __future__ import annotations

from collections import OrderedDict

import awkward as ak

from ._fusion import FusedNode, _children, fuse
from ._ir import (
    BinaryOpNode,
    CombinationsNode,
    ComparisonNode,
    ConstantNode,
    FilterNode,
    GetItemNode,
    InputNode,
    OpNode,
    OpType,
    ReduceNode,
    SelectListsNode,
)
from ._nvtx import nvtx

# Sentinel: no backend fused-kernel path was taken, fall back to eager eval.
_NO_FUSED_KERNEL = object()

# Axis=-1 reductions used both by ReduceNode and the fused reduction stage.
_REDUCE_OPS = {
    OpType.SUM: lambda a: ak.sum(a, axis=-1),
    OpType.MEAN: lambda a: ak.mean(a, axis=-1),
    OpType.MAX: lambda a: ak.max(a, axis=-1),
    OpType.MIN: lambda a: ak.min(a, axis=-1),
}

_BINARY_OPS = {
    OpType.ADD: lambda a, b: a + b,
    OpType.SUB: lambda a, b: a - b,
    OpType.MUL: lambda a, b: a * b,
    OpType.DIV: lambda a, b: a / b,
    OpType.POW: lambda a, b: a**b,
    OpType.MOD: lambda a, b: a % b,
    OpType.FLOORDIV: lambda a, b: a // b,
}

_COMPARISON_OPS = {
    OpType.LT: lambda a, b: a < b,
    OpType.LE: lambda a, b: a <= b,
    OpType.GT: lambda a, b: a > b,
    OpType.GE: lambda a, b: a >= b,
    OpType.EQ: lambda a, b: a == b,
    OpType.NE: lambda a, b: a != b,
}


def _uniform_backend(values):
    """Return the single backend name shared by all array leaves, else ``None``.

    Scalars (no ``.layout``) are ignored.  ``None`` means the leaves mix
    backends (or there are no array leaves) — fusion must decline so the eager
    path decides validity (e.g. a cuda+cpu mix raises ``ValueError`` just like
    ``fuse=False``, instead of silently copying one side to the other device).
    """
    names = set()
    for value in values:
        backend = getattr(getattr(value, "layout", None), "backend", None)
        name = getattr(backend, "name", None)
        if name is not None:
            names.add(name)
    return names.pop() if len(names) == 1 else None


class IRExecutor:
    """
    Executes the IR by traversing the graph and dispatching each node to
    the corresponding (backend-dispatched) Awkward operation.

    Memoized results live in an LRU cache bounded by both entry count
    (`max_entries`) and total result bytes (`max_bytes`); either can be
    None for no limit.  A result larger than `max_bytes` on its own is not
    cached at all.  LRU rather than refcount-based eviction because future
    consumers of a node are unknowable at eviction time, and eviction is
    always safe: the owning ``LazyAwkwardArray`` caches its own root result,
    so evicting an intermediate only risks recomputation, never breakage.
    ``InputNode``/``ConstantNode`` results are never cached — returning the
    stored reference is already free.
    """

    DEFAULT_MAX_ENTRIES = 64
    DEFAULT_MAX_BYTES = 256 * 1024**2  # 256 MiB

    def __init__(self, max_entries=DEFAULT_MAX_ENTRIES, max_bytes=DEFAULT_MAX_BYTES):
        self._memo = OrderedDict()  # node_id -> result, LRU order
        self._memo_sizes = {}  # node_id -> estimated bytes
        self._memo_bytes = 0
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        # Which path each FusedNode took, so callers (benchmarks, tests) can
        # confirm a real fused kernel ran instead of a silent eager fallback.
        self.fused_hits = {"cuda": 0, "cpu": 0, "eager": 0}

    def invalidate(self):
        """Clear all memoized node results."""
        self._memo.clear()
        self._memo_sizes.clear()
        self._memo_bytes = 0

    @staticmethod
    def _result_nbytes(result):
        """Estimated size of a result; 0 for objects without nbytes."""
        try:
            return int(result.nbytes)
        except (AttributeError, TypeError):
            return 0

    def _memo_insert(self, key, result):
        size = self._result_nbytes(result)
        if self.max_bytes is not None and size > self.max_bytes:
            return  # would immediately evict everything else; skip caching
        self._memo[key] = result
        self._memo_sizes[key] = size
        self._memo_bytes += size
        while (self.max_entries is not None and len(self._memo) > self.max_entries) or (
            self.max_bytes is not None and self._memo_bytes > self.max_bytes
        ):
            evicted_key, _ = self._memo.popitem(last=False)
            self._memo_bytes -= self._memo_sizes.pop(evicted_key)

    @nvtx.annotate("compile_and_execute")
    def compile_and_execute(self, node: OpNode) -> ak.Array:
        """Run the fusion pass, then execute the fused graph.

        This is the entry point that realizes the *dynamic compilation* step:
        element-wise regions of the expression graph are collapsed into
        ``FusedNode``s (one kernel each on a fusing backend) before execution.
        ``execute`` remains the plain interpreter over whatever graph it is
        handed, so passing the original (unfused) root gives the no-fuse
        debug path.
        """
        return self.execute(self.optimize(node))

    @nvtx.annotate("execute_ir")
    def execute(self, node: OpNode) -> ak.Array:
        """Execute an IR node and return the result.

        Evaluation is an explicit-stack postorder walk (no recursion, so
        loop-built graphs thousands of nodes deep execute fine).  Sub-graphs
        below a memoized node are pruned — a memo hit never recomputes its
        dependencies.  ``results`` holds this call's realized intermediates;
        they are released when the call returns (the bounded LRU memo decides
        what survives across calls).
        """
        results: dict[int, object] = {}
        stack: list[tuple[OpNode, bool]] = [(node, False)]
        while stack:
            current, processed = stack.pop()
            nid = current.node_id
            if nid in results:
                continue
            if processed:
                value = self._execute_node(current, results)
                self._memo_insert(nid, value)
                results[nid] = value
                continue
            # Leaves are free to "recompute" (they return a stored reference);
            # keeping them out of the memo preserves the budget for real work.
            if isinstance(current, (ConstantNode, InputNode)):
                results[nid] = self._execute_node(current, results)
                continue
            if nid in self._memo:
                self._memo.move_to_end(nid)
                results[nid] = self._memo[nid]
                continue
            stack.append((current, True))
            for child in _children(current):
                if child.node_id not in results:
                    stack.append((child, False))
        return results[node.node_id]

    def _execute_node(self, node: OpNode, results: dict) -> ak.Array:
        """Execute a single IR node; children are looked up in ``results``."""
        if isinstance(node, InputNode):
            return node.get_array()

        elif isinstance(node, ConstantNode):
            return node.value

        elif isinstance(node, BinaryOpNode):
            left = results[node.left.node_id]
            right = results[node.right.node_id]
            return _BINARY_OPS[node.op_type](left, right)

        elif isinstance(node, ComparisonNode):
            left = results[node.left.node_id]
            right = results[node.right.node_id]
            return _COMPARISON_OPS[node.op_type](left, right)

        elif isinstance(node, FilterNode):
            # Boolean-mask ``__getitem__`` dispatches to the array's backend,
            # so this is GPU-accelerated for cuda-backed arrays already.
            return results[node.input.node_id][results[node.condition.node_id]]

        elif isinstance(node, SelectListsNode):
            return results[node.input.node_id][results[node.mask.node_id]]

        elif isinstance(node, ReduceNode):
            return _REDUCE_OPS[node.reduce_op](results[node.input.node_id])

        elif isinstance(node, CombinationsNode):
            return ak.combinations(
                results[node.input.node_id],
                node.n,
                replacement=node.replacement,
                axis=node.axis,
                fields=node.fields,
            )

        elif isinstance(node, GetItemNode):
            key = node.key
            if isinstance(key, OpNode):
                key = results[key.node_id]
            return results[node.input.node_id][key]

        elif isinstance(node, FusedNode):
            return self._execute_fused(node, results)

        else:
            raise NotImplementedError(f"Execution not implemented for {type(node)}")

    def _execute_fused(self, node: FusedNode, results: dict) -> ak.Array:
        """Execute a fused element-wise (optionally +reduction) region.

        Leaves were realized first (memoized like any other node).  A backend
        fusion codegen may emit the region as a single kernel; otherwise the
        fused expression is evaluated eagerly in one pass.  In every case a
        terminating reduction is produced by the eager Awkward reducer
        (``_REDUCE_OPS``) — on CUDA the codegen fuses ``sum`` into its kernel
        and returns the final result directly; the CPU codegen and the eager
        fallback fuse only the element-wise map and let ``_REDUCE_OPS`` apply
        the reduction, so the fused and interpreter paths return the identical
        Awkward-typed result (same dtype, masking, and empty-list semantics).
        """
        values = [results[leaf.node_id] for leaf in node.leaves]

        # CUDA: one kernel for the map (and the sum reduction, when fusible).
        fused = self._maybe_fused_kernel(node, values)
        if fused is not _NO_FUSED_KERNEL:
            self.fused_hits["cuda"] += 1
            return fused

        # CPU: fuse the element-wise map into one flat pass; the reduction is
        # delegated to the eager reducer below for identical result semantics.
        mapped = self._maybe_cpu_fused(node, values)
        if mapped is not _NO_FUSED_KERNEL:
            self.fused_hits["cpu"] += 1
            if node.reduce_op is not None:
                return _REDUCE_OPS[node.reduce_op](mapped)
            return mapped

        # Eager fallback: composite expression, then the eager reduction.
        self.fused_hits["eager"] += 1
        result = node.evaluate(values)
        if node.reduce_op is not None:
            result = _REDUCE_OPS[node.reduce_op](result)
        return result

    @staticmethod
    def _maybe_fused_kernel(node: FusedNode, values):
        """Try to emit a single backend kernel for the region.

        Returns the result, or ``_NO_FUSED_KERNEL`` if no codegen applies.
        Requires every array leaf to be cuda-backed: a mixed cpu/cuda region
        must fall back so it fails (or not) exactly like the eager expression
        instead of silently migrating data to the device.  Any exception from
        the codegen also falls back — fusion is a fast path, never a
        correctness dependency, so a codegen defect must degrade to the eager
        (still backend-dispatched, still correct) evaluation, not fail the
        computation.
        """
        if _uniform_backend(values) != "cuda":
            return _NO_FUSED_KERNEL
        try:
            from awkward._connect.cuda._fusion_codegen import execute_fused_cuda
        except ImportError:
            return _NO_FUSED_KERNEL
        try:
            return execute_fused_cuda(node, values)
        except Exception:
            return _NO_FUSED_KERNEL

    @staticmethod
    def _maybe_cpu_fused(node: FusedNode, values):
        """Try to evaluate the region as one flat-buffer NumPy pass.

        Collapses the per-op ``ak`` dispatch of an element-wise chain into a
        single pass over the shared content buffer.  Returns the result or
        ``_NO_FUSED_KERNEL`` when the region shape is unsupported or the
        codegen fails for any reason (same fallback discipline as the CUDA
        path).
        """
        if _uniform_backend(values) != "cpu":
            return _NO_FUSED_KERNEL
        try:
            from awkward._connect.cpu._fusion_codegen import execute_fused_cpu
        except ImportError:
            return _NO_FUSED_KERNEL
        try:
            return execute_fused_cpu(node, values)
        except Exception:
            return _NO_FUSED_KERNEL

    def visualize_ir(self, node: OpNode, indent: int = 0) -> str:
        """Generate a string visualization of the IR tree"""
        prefix = "  " * indent
        lines = [f"{prefix}{node}"]

        if isinstance(node, (BinaryOpNode, ComparisonNode)):
            lines.append(self.visualize_ir(node.left, indent + 1))
            lines.append(self.visualize_ir(node.right, indent + 1))
        elif isinstance(node, (FilterNode, SelectListsNode)):
            lines.append(f"{prefix}  Input:")
            lines.append(self.visualize_ir(node.input, indent + 2))
            lines.append(f"{prefix}  Condition/Mask:")
            if hasattr(node, "condition"):
                lines.append(self.visualize_ir(node.condition, indent + 2))
            else:
                lines.append(self.visualize_ir(node.mask, indent + 2))
        elif isinstance(node, ReduceNode):
            lines.append(self.visualize_ir(node.input, indent + 1))
        elif isinstance(node, CombinationsNode):
            lines.append(
                f"{prefix}  n={node.n}, replacement={node.replacement}, axis={node.axis}"
            )
            lines.append(self.visualize_ir(node.input, indent + 1))
        elif isinstance(node, GetItemNode):
            lines.append(f"{prefix}  key={node.key!r}")
            lines.append(self.visualize_ir(node.input, indent + 1))
            if isinstance(node.key, OpNode):
                lines.append(self.visualize_ir(node.key, indent + 1))
        elif isinstance(node, FusedNode):
            red = f" -> reduce {node.reduce_op.value}" if node.reduce_op else ""
            lines.append(f"{prefix}  expr={node.expr_text}{red}")
            for i, leaf in enumerate(node.leaves):
                lines.append(f"{prefix}  ${i}:")
                lines.append(self.visualize_ir(leaf, indent + 2))

        return "\n".join(lines)

    def optimize(self, node: OpNode) -> OpNode:
        """
        Apply optimization passes to the IR and return the rewritten root.

        Currently one pass runs: **kernel fusion** (``_fusion.fuse``), which
        collapses maximal element-wise regions (and transform+reduce) into
        single ``FusedNode`` kernels.  The pass also folds shared
        sub-expressions to one materialization point (single-use fusion), so
        common-subexpression elimination for the fused regions falls out of it.

        The pass is idempotent: a graph that is already fused (its element-wise
        regions are ``FusedNode``s) is returned unchanged, because ``FusedNode``
        is opaque to ``fuse`` and every other node is rebuilt structurally.

        Future passes (constant folding, dead-code elimination) compose here.
        """
        return fuse(node)
