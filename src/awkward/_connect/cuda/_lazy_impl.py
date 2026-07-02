# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import cupy as cp
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
    IRNode,
    OpType,
    ReduceNode,
    SelectListsNode,
)


class LazyAwkwardArray:
    """
    Wrapper around Awkward Array that builds an IR for delayed computation.

    This allows chaining operations without immediate execution, enabling
    optimization and fusion of GPU kernels.
    """

    def __init__(self, ir_node: IRNode, executor: IRExecutor | None = None):
        self.ir_node = ir_node
        self.executor = executor or IRExecutor()
        self._computed_result = None

    @classmethod
    def from_array(cls, array: ak.Array, executor: IRExecutor | None = None):
        """Create a LazyAwkwardArray from an existing Awkward Array"""
        input_node = InputNode(array)
        return cls(input_node, executor)

    def compute(self) -> ak.Array:
        """Execute the IR and return the result"""
        if self._computed_result is None:
            self._computed_result = self.executor.execute(self.ir_node)
        return self._computed_result

    def _invalidate_cache(self):
        """Invalidate cached computation result"""
        self._computed_result = None

    # Arithmetic operations
    def __add__(self, other):
        self._invalidate_cache()
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.ADD, self.ir_node, other_node), self.executor
        )

    def __sub__(self, other):
        self._invalidate_cache()
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.SUB, self.ir_node, other_node), self.executor
        )

    def __mul__(self, other):
        self._invalidate_cache()
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.MUL, self.ir_node, other_node), self.executor
        )

    def __truediv__(self, other):
        self._invalidate_cache()
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.DIV, self.ir_node, other_node), self.executor
        )

    def __pow__(self, other):
        self._invalidate_cache()
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.POW, self.ir_node, other_node), self.executor
        )

    # Comparison operations
    def __lt__(self, other):
        self._invalidate_cache()
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            ComparisonNode(OpType.LT, self.ir_node, other_node), self.executor
        )

    def __le__(self, other):
        self._invalidate_cache()
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            ComparisonNode(OpType.LE, self.ir_node, other_node), self.executor
        )

    def __gt__(self, other):
        self._invalidate_cache()
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            ComparisonNode(OpType.GT, self.ir_node, other_node), self.executor
        )

    def __ge__(self, other):
        self._invalidate_cache()
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            ComparisonNode(OpType.GE, self.ir_node, other_node), self.executor
        )

    def __eq__(self, other):
        self._invalidate_cache()
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            ComparisonNode(OpType.EQ, self.ir_node, other_node), self.executor
        )

    def __ne__(self, other):
        self._invalidate_cache()
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            ComparisonNode(OpType.NE, self.ir_node, other_node), self.executor
        )

    # Right-hand operations
    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        self._invalidate_cache()
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.SUB, other_node, self.ir_node), self.executor
        )

    def __rtruediv__(self, other):
        self._invalidate_cache()
        other_node = self._to_ir_node(other)
        return LazyAwkwardArray(
            BinaryOpNode(OpType.DIV, other_node, self.ir_node), self.executor
        )

    # List operations
    def filter(self, condition):
        """Filter elements in lists based on a condition"""
        self._invalidate_cache()
        if isinstance(condition, LazyAwkwardArray):
            cond_node = condition.ir_node
        else:
            cond_node = self._to_ir_node(condition)

        return LazyAwkwardArray(FilterNode(self.ir_node, cond_node), self.executor)

    def select_lists(self, mask):
        """Select entire lists based on a boolean mask"""
        self._invalidate_cache()
        mask_node = self._to_ir_node(mask)
        return LazyAwkwardArray(SelectListsNode(self.ir_node, mask_node), self.executor)

    def sum(self):
        """Sum reduction over lists"""
        self._invalidate_cache()
        return LazyAwkwardArray(ReduceNode(self.ir_node, OpType.SUM), self.executor)

    def mean(self):
        """Mean reduction over lists"""
        self._invalidate_cache()
        return LazyAwkwardArray(ReduceNode(self.ir_node, OpType.MEAN), self.executor)

    def max(self):
        """Max reduction over lists"""
        self._invalidate_cache()
        return LazyAwkwardArray(ReduceNode(self.ir_node, OpType.MAX), self.executor)

    def min(self):
        """Min reduction over lists"""
        self._invalidate_cache()
        return LazyAwkwardArray(ReduceNode(self.ir_node, OpType.MIN), self.executor)

    def combinations(
        self,
        n: int,
        replacement: bool = False,
        axis: int = 1,
        fields: list | None = None,
    ):
        """
        Generate n-element combinations from lists.

        Args:
            n: Number of elements in each combination
            replacement: If True, allow replacement (combinations with repetition)
            axis: Axis along which to generate combinations (default: 1)
            fields: Optional field names for the combination tuple
        """
        self._invalidate_cache()
        return LazyAwkwardArray(
            CombinationsNode(self.ir_node, n, replacement, axis, fields), self.executor
        )

    def __getitem__(self, key: str | int | slice):
        """
        Access fields or indices lazily.
        """
        self._invalidate_cache()
        return LazyAwkwardArray(GetItemNode(self.ir_node, key), self.executor)

    def _to_ir_node(self, value):
        """Convert a value to an IR node"""
        if isinstance(value, LazyAwkwardArray):
            return value.ir_node
        elif isinstance(value, ak.Array):
            return InputNode(value)
        elif isinstance(value, (int, float, np.ndarray, cp.ndarray)):
            return ConstantNode(value)
        else:
            raise TypeError(f"Cannot convert {type(value)} to IR node")

    def visualize(self) -> str:
        """Generate a string representation of the IR"""
        return self.executor.visualize_ir(self.ir_node)

    def __repr__(self):
        """String representation showing this is a lazy array"""
        return f"LazyAwkwardArray(op={self.ir_node.op_type.value}, node_id={self.ir_node.node_id})"


def lazy(array: ak.Array) -> LazyAwkwardArray:
    """
    Convert an Awkward Array to a lazy evaluation wrapper.

    Example:
        >>> import awkward as ak
        >>>
        >>> # Create lazy array
        >>> arr = ak.Array([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
        >>> lazy_arr = ak.cuda.lazy(arr)
        >>>
        >>> # Build computation graph
        >>> result = lazy_arr * 2 + 1
        >>> filtered = result.filter(result > 5)
        >>>
        >>> # Execute when needed
        >>> output = filtered.compute()
    """
    return LazyAwkwardArray.from_array(array)
