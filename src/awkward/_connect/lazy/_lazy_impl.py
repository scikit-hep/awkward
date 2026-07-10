# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak

from ._executor import IRExecutor
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

try:
    import cupy as cp
except ImportError:
    cp = None


class LazyAwkwardArray:
    """
    Wrapper around Awkward Array that builds an IR for delayed computation.

    This allows chaining operations without immediate execution, enabling
    optimization and fusion of kernels.
    """

    def __init__(self, ir_node: OpNode, executor: IRExecutor | None = None):
        self.ir_node = ir_node
        self.executor = executor or IRExecutor()
        self._computed_result = None
        self._computed_fuse = None

    @classmethod
    def from_array(cls, array: ak.Array, executor: IRExecutor | None = None):
        """Create a ``LazyAwkwardArray`` from an existing Awkward Array.

        Args:
            array (ak.Array): The array to wrap as an ``InputNode`` leaf.
            executor (IRExecutor or None): Executor to reuse; a fresh one is
                created when ``None``.

        Returns a new :class:`LazyAwkwardArray`.
        """
        input_node = InputNode(array)
        return cls(input_node, executor)

    def compute(self, fuse: bool = True) -> ak.Array:
        """Execute the IR and return the result.

        Args:
            fuse (bool): With ``True`` (default) the expression graph is compiled
                first, collapsing element-wise regions into single fused kernels
                (:func:`awkward._connect.lazy._fusion.fuse`); ``False`` runs the
                plain per-node interpreter (the no-fuse / debug path, which keeps
                every intermediate visible). Both are numerically identical.

        Returns the computed ``ak.Array`` (or reduction result).
        """
        if self._computed_result is None or self._computed_fuse != fuse:
            if fuse:
                self._computed_result = self.executor.compile_and_execute(self.ir_node)
            else:
                self._computed_result = self.executor.execute(self.ir_node)
            self._computed_fuse = fuse
        return self._computed_result

    def compile(self) -> OpNode:
        """Compile the graph without executing it.

        Useful for inspecting how many kernels the computation will launch (see
        :meth:`fusion_stats`) or for debugging the fusion pass.

        Returns the fused expression graph as an ``OpNode``.
        """
        from ._fusion import fuse as _fuse

        return _fuse(self.ir_node)

    def fusion_stats(self) -> dict:
        """Report how fusion reshapes this graph.

        Returns the dict from :func:`awkward._connect.lazy._fusion.fusion_stats`
        (``elementwise_before``, ``fused_regions``, ``materialized``).
        """
        from ._fusion import fusion_stats as _stats

        return _stats(self.ir_node)

    def invalidate(self):
        """Discard cached results (this wrapper's and the executor's memo).

        Call this if an input array was mutated in place.
        """
        self._computed_result = None
        self._computed_fuse = None
        self.executor.invalidate()

    # Arithmetic operations
    def __add__(self, other):
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.ADD, self.ir_node, other_node), self.executor
        )

    def __sub__(self, other):
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.SUB, self.ir_node, other_node), self.executor
        )

    def __mul__(self, other):
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.MUL, self.ir_node, other_node), self.executor
        )

    def __truediv__(self, other):
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.DIV, self.ir_node, other_node), self.executor
        )

    def __pow__(self, other):
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.POW, self.ir_node, other_node), self.executor
        )

    def __mod__(self, other):
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.MOD, self.ir_node, other_node), self.executor
        )

    def __floordiv__(self, other):
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.FLOORDIV, self.ir_node, other_node), self.executor
        )

    def __neg__(self):
        # 0 - x follows the same value-based promotion as eager -x.
        return LazyAwkwardArray(
            BinaryOpNode(OpType.SUB, ConstantNode(0), self.ir_node), self.executor
        )

    # Comparison operations
    def __lt__(self, other):
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            ComparisonNode(OpType.LT, self.ir_node, other_node), self.executor
        )

    def __le__(self, other):
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            ComparisonNode(OpType.LE, self.ir_node, other_node), self.executor
        )

    def __gt__(self, other):
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            ComparisonNode(OpType.GT, self.ir_node, other_node), self.executor
        )

    def __ge__(self, other):
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            ComparisonNode(OpType.GE, self.ir_node, other_node), self.executor
        )

    def __eq__(self, other):
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            ComparisonNode(OpType.EQ, self.ir_node, other_node), self.executor
        )

    def __ne__(self, other):
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            ComparisonNode(OpType.NE, self.ir_node, other_node), self.executor
        )

    # __eq__ returns a lazy comparison, so (like NumPy arrays) instances are
    # unhashable and must not be coerced to bool implicitly.
    __hash__ = None

    def __bool__(self):
        raise TypeError(
            "the truth value of a LazyAwkwardArray is ambiguous; "
            "call .compute() and use ak.all/ak.any on the result"
        )

    # Right-hand operations
    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.SUB, other_node, self.ir_node), self.executor
        )

    def __rtruediv__(self, other):
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.DIV, other_node, self.ir_node), self.executor
        )

    def __rpow__(self, other):
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.POW, other_node, self.ir_node), self.executor
        )

    def __rmod__(self, other):
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.MOD, other_node, self.ir_node), self.executor
        )

    def __rfloordiv__(self, other):
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.FLOORDIV, other_node, self.ir_node), self.executor
        )

    # List operations
    def filter(self, condition):
        """Filter elements within lists by a condition.

        Args:
            condition: A boolean mask (a ``LazyAwkwardArray``, ``ak.Array``, or
                array-like) selecting elements to keep within each list.

        Returns a new ``LazyAwkwardArray`` wrapping the filter operation.
        """
        cond_node = self._to_ir_node(condition)
        return LazyAwkwardArray(FilterNode(self.ir_node, cond_node), self.executor)

    def select_lists(self, mask):
        """Select entire lists by a per-list boolean mask.

        Args:
            mask: A per-list boolean mask.

        Returns a new ``LazyAwkwardArray`` wrapping the select-lists operation.
        """
        mask_node = self._to_ir_node(mask)
        return LazyAwkwardArray(SelectListsNode(self.ir_node, mask_node), self.executor)

    def sum(self):
        """Returns a new ``LazyAwkwardArray`` for a per-list sum reduction."""
        return LazyAwkwardArray(ReduceNode(self.ir_node, OpType.SUM), self.executor)

    def mean(self):
        """Returns a new ``LazyAwkwardArray`` for a per-list mean reduction."""
        return LazyAwkwardArray(ReduceNode(self.ir_node, OpType.MEAN), self.executor)

    def max(self):
        """Returns a new ``LazyAwkwardArray`` for a per-list max reduction."""
        return LazyAwkwardArray(ReduceNode(self.ir_node, OpType.MAX), self.executor)

    def min(self):
        """Returns a new ``LazyAwkwardArray`` for a per-list min reduction."""
        return LazyAwkwardArray(ReduceNode(self.ir_node, OpType.MIN), self.executor)

    def combinations(
        self,
        n: int,
        replacement: bool = False,
        axis: int = 1,
        fields: list | None = None,
    ):
        """Generate n-element combinations from lists.

        Args:
            n (int): Number of elements in each combination.
            replacement (bool): If True, allow replacement (combinations with
                repetition).
            axis (int): Axis along which to generate combinations.
            fields (list or None): Optional field names for the combination
                tuple.

        Returns a new ``LazyAwkwardArray`` wrapping the combinations operation.
        """
        return LazyAwkwardArray(
            CombinationsNode(self.ir_node, n, replacement, axis, fields), self.executor
        )

    def __getitem__(self, key):
        """Access fields, indices, or slices lazily.

        A lazy (or eager) array key selects like eager ``__getitem__``, so
        ``lazy_arr[lazy_arr > 5]`` matches ``arr[arr > 5]``.

        Args:
            key: A field name, index, slice, or a (lazy or eager) array key.

        Returns a new ``LazyAwkwardArray`` wrapping the item access.
        """
        if isinstance(key, (LazyAwkwardArray, ak.Array)):
            key = self._to_ir_node(key)
        return LazyAwkwardArray(GetItemNode(self.ir_node, key), self.executor)

    def _to_ir_node(self, value):
        """Convert a value to an IR node.

        Args:
            value: A ``LazyAwkwardArray``, ``ak.Array``, or numeric scalar/array.

        Returns the corresponding ``OpNode`` (``InputNode`` / ``ConstantNode``,
        or the wrapped node for a ``LazyAwkwardArray``).

        Raises:
            TypeError: If ``value`` cannot be converted to an IR node.
        """
        if isinstance(value, LazyAwkwardArray):
            return value.ir_node
        elif isinstance(value, ak.Array):
            return InputNode(value)
        elif isinstance(value, (int, float, np.ndarray)) or (
            cp is not None and isinstance(value, cp.ndarray)
        ):
            return ConstantNode(value)
        else:
            raise TypeError(f"Cannot convert {type(value)} to IR node")

    def visualize(self, fused: bool = False) -> str:
        """Generate a string representation of the IR.

        Args:
            fused (bool): If True, show the fused (compiled) graph so the
                ``FusedNode`` regions and their leaves are visible; otherwise
                show the original operator graph.

        Returns the IR tree as a multi-line string.
        """
        node = self.compile() if fused else self.ir_node
        return self.executor.visualize_ir(node)

    def __repr__(self):
        """String representation showing this is a lazy array"""
        return f"LazyAwkwardArray(op={self.ir_node.op_type.value}, node_id={self.ir_node.node_id})"


def lazy(array: ak.Array) -> LazyAwkwardArray:
    """Convert an Awkward Array to a lazy evaluation wrapper.

    Args:
        array (ak.Array): The array to wrap for delayed, fused execution.

    Returns a :class:`LazyAwkwardArray` over ``array``.

    Example:
        >>> import awkward as ak
        >>> from awkward._connect import cpu   # internal for now (PoC)
        >>> arr = ak.Array([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
        >>> lazy_arr = cpu.lazy(arr)   # or awkward._connect.cuda.lazy for cuda
        >>> result = lazy_arr * 2 + 1
        >>> filtered = result.filter(result > 5)
        >>> output = filtered.compute()
    """
    return LazyAwkwardArray.from_array(array)
