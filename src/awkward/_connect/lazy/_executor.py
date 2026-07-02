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

from ._ir import (
    BinaryOpNode,
    CombinationsNode,
    ComparisonNode,
    ConstantNode,
    FilterNode,
    GetItemNode,
    InputNode,
    IRNode,
    OpType,
    ReduceNode,
    SelectListsNode,
)

try:
    import nvtx
except ImportError:

    class nvtx:
        @staticmethod
        def annotate(*args, **kwargs):
            def deco(fn):
                return fn

            return deco


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

    @nvtx.annotate("execute_ir")
    def execute(self, node: IRNode) -> ak.Array:
        """Execute an IR node and return the result"""
        # Leaves are free to "recompute" (they return a stored reference);
        # keeping them out of the memo preserves the budget for real work.
        if isinstance(node, (ConstantNode, InputNode)):
            return self._execute_node(node)

        # Check cache first
        if node.node_id in self._memo:
            self._memo.move_to_end(node.node_id)
            return self._memo[node.node_id]

        result = self._execute_node(node)
        self._memo_insert(node.node_id, result)
        return result

    def _execute_node(self, node: IRNode) -> ak.Array:
        """Execute a single IR node"""
        if isinstance(node, InputNode):
            return node.get_array()

        elif isinstance(node, ConstantNode):
            return node.value

        elif isinstance(node, BinaryOpNode):
            return self._execute_binary_op(node)

        elif isinstance(node, ComparisonNode):
            return self._execute_comparison(node)

        elif isinstance(node, FilterNode):
            return self._execute_filter(node)

        elif isinstance(node, SelectListsNode):
            return self._execute_select_lists(node)

        elif isinstance(node, ReduceNode):
            return self._execute_reduce(node)

        elif isinstance(node, CombinationsNode):
            return self._execute_combinations(node)

        elif isinstance(node, GetItemNode):
            return self._execute_getitem(node)

        else:
            raise NotImplementedError(f"Execution not implemented for {type(node)}")

    def _execute_binary_op(self, node: BinaryOpNode) -> ak.Array:
        """Execute binary operations"""
        left = self.execute(node.left)
        right = self.execute(node.right)

        op_map = {
            OpType.ADD: lambda a, b: a + b,
            OpType.SUB: lambda a, b: a - b,
            OpType.MUL: lambda a, b: a * b,
            OpType.DIV: lambda a, b: a / b,
            OpType.POW: lambda a, b: a**b,
        }

        return op_map[node.op_type](left, right)

    def _execute_comparison(self, node: ComparisonNode) -> ak.Array:
        """Execute comparison operations"""
        left = self.execute(node.left)
        right = self.execute(node.right)

        op_map = {
            OpType.LT: lambda a, b: a < b,
            OpType.LE: lambda a, b: a <= b,
            OpType.GT: lambda a, b: a > b,
            OpType.GE: lambda a, b: a >= b,
            OpType.EQ: lambda a, b: a == b,
            OpType.NE: lambda a, b: a != b,
        }

        return op_map[node.op_type](left, right)

    def _execute_filter(self, node: FilterNode) -> ak.Array:
        """Execute filter with a materialized boolean condition.

        Boolean-mask ``__getitem__`` dispatches to the array's backend, so
        this is GPU-accelerated for cuda-backed arrays already.
        """
        input_array = self.execute(node.input)
        condition = self.execute(node.condition)
        return input_array[condition]

    def _execute_select_lists(self, node: SelectListsNode) -> ak.Array:
        """Execute select-lists with a materialized per-list mask."""
        input_array = self.execute(node.input)
        mask = self.execute(node.mask)
        return input_array[mask]

    def _execute_reduce(self, node: ReduceNode) -> ak.Array:
        """Execute reduction operations"""
        input_array = self.execute(node.input)

        # Use Awkward's built-in reductions
        op_map = {
            OpType.SUM: lambda a: ak.sum(a, axis=-1),
            OpType.MEAN: lambda a: ak.mean(a, axis=-1),
            OpType.MAX: lambda a: ak.max(a, axis=-1),
            OpType.MIN: lambda a: ak.min(a, axis=-1),
        }

        return op_map[node.reduce_op](input_array)

    def _execute_combinations(self, node: CombinationsNode) -> ak.Array:
        """Execute combinations operation"""
        input_array = self.execute(node.input)

        return ak.combinations(
            input_array,
            node.n,
            replacement=node.replacement,
            axis=node.axis,
            fields=node.fields,
        )

    def _execute_getitem(self, node: GetItemNode) -> ak.Array:
        """Execute field/index access"""
        input_array = self.execute(node.input)
        return input_array[node.key]

    def visualize_ir(self, node: IRNode, indent: int = 0) -> str:
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

        return "\n".join(lines)

    def optimize(self, node: IRNode) -> IRNode:
        """
        Apply optimization passes to the IR.

        Potential optimizations:
        - Constant folding
        - Common subexpression elimination
        - Kernel fusion
        - Dead code elimination
        """
        # Placeholder for optimization passes
        return node
