# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Operator-level IR built by ``LazyAwkwardArray`` (backend-neutral).

The base class is ``OpNode`` — distinct from ``awkward._connect.lazy.core.IRNode``,
which is the low-level compute/lower protocol used by the backend-specific
``ir_nodes`` modules.  The two systems are independent.
"""

from __future__ import annotations

import itertools
from enum import Enum

import numpy as np

import awkward as ak

try:
    import cupy as cp
except ImportError:
    cp = None

# ============================================================================
# IR Node Definitions
# ============================================================================


class OpType(Enum):
    """Operation types in the IR"""

    # Element-wise operations
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"
    MOD = "mod"
    FLOORDIV = "floordiv"

    # Comparison operations
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    EQ = "eq"
    NE = "ne"

    # List operations
    FILTER = "filter"
    REDUCE = "reduce"
    SELECT_LISTS = "select_lists"
    COMBINATIONS = "combinations"
    GETITEM = "getitem"

    # Aggregations
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"

    # Data operations
    INPUT = "input"
    CONSTANT = "constant"

    # Compiler-internal: a region of element-wise ops (optionally terminated
    # by a reduction) collapsed into a single fused kernel by the fusion pass.
    FUSED = "fused"


class OpNode:
    """Base class for operator-level IR nodes"""

    # itertools.count is atomic in CPython, so node ids are thread-safe
    _id_counter = itertools.count(1)

    def __init__(self, op_type: OpType, dtype=None, shape=None):
        self.op_type = op_type
        self.dtype = dtype
        self.shape = shape
        self.node_id = next(OpNode._id_counter)

    def __hash__(self):
        return self.node_id

    def __eq__(self, other):
        """Equality based on node identity, not value"""
        if not isinstance(other, OpNode):
            return False
        return self.node_id == other.node_id

    def __repr__(self):
        return f"{self.__class__.__name__}(op={self.op_type.value}, id={self.node_id})"


def _leaf_numpy_dtype(array):
    """Return the leaf ``NumpyArray`` dtype of ``array``, or ``None``.

    Descends through option/index/list wrappers to the numeric leaf; returns
    ``None`` for record arrays (no single dtype) or anything without one.
    """
    try:
        layout = array.layout
    except AttributeError:
        return None
    # Records expose ``contents`` (plural), not a single ``content``; stopping
    # there yields None, which is the right answer (no single dtype).
    while hasattr(layout, "content"):
        layout = layout.content
    dtype = getattr(layout, "dtype", None)
    return np.dtype(dtype) if dtype is not None else None


class InputNode(OpNode):
    """Represents an input array.

    Holds a strong reference: the whole point of lazy evaluation is that
    `compute()` may run long after graph construction, so the graph must
    keep its leaves alive.
    """

    def __init__(self, array: ak.Array):
        super().__init__(op_type=OpType.INPUT)
        self.array = array
        if isinstance(array, ak.Array):
            # Infer the leaf numeric dtype so downstream binary/reduce nodes
            # propagate it (e.g. an integer input must not become float64 in
            # the fused output allocation).
            self.dtype = _leaf_numpy_dtype(array)

    def get_array(self):
        """Get the referenced array."""
        if self.array is None:
            raise ValueError("No array reference available")
        return self.array


class ConstantNode(OpNode):
    """Represents a constant value"""

    def __init__(self, value):
        dtype = None
        if isinstance(value, (int, float)):
            dtype = np.dtype(type(value))
        elif hasattr(value, "dtype"):
            dtype = value.dtype
        super().__init__(op_type=OpType.CONSTANT, dtype=dtype)

        if isinstance(value, (int, float, np.ndarray)) or (
            cp is not None and isinstance(value, cp.ndarray)
        ):
            self.value = value
        else:
            # Try to convert to numpy
            self.value = np.asarray(value)


class BinaryOpNode(OpNode):
    """Represents binary operations (add, mul, etc.)"""

    def __init__(self, op_type: OpType, left: OpNode, right: OpNode):
        super().__init__(op_type=op_type)
        self.left = left
        self.right = right

        # Infer dtype from operands if possible
        if left.dtype is not None and right.dtype is not None:
            # Use numpy's type promotion rules
            self.dtype = np.result_type(left.dtype, right.dtype)
            # True division always produces an inexact type
            if op_type is OpType.DIV and not np.issubdtype(self.dtype, np.inexact):
                self.dtype = np.result_type(self.dtype, np.float64)


class ComparisonNode(OpNode):
    """Represents comparison operations"""

    def __init__(self, op_type: OpType, left: OpNode, right: OpNode):
        super().__init__(op_type=op_type, dtype=np.bool_)
        self.left = left
        self.right = right


class FilterNode(OpNode):
    """Represents filtering operations on lists"""

    def __init__(self, input: OpNode, condition: OpNode):
        super().__init__(op_type=OpType.FILTER)
        self.input = input
        self.condition = condition
        # Output dtype matches input dtype
        self.dtype = input.dtype if hasattr(input, "dtype") else None


class ReduceNode(OpNode):
    """Represents reduction operations on lists"""

    def __init__(self, input: OpNode, reduce_op: OpType):
        super().__init__(op_type=OpType.REDUCE)
        self.input = input
        self.reduce_op = reduce_op
        # Reductions typically preserve dtype except for MEAN
        if reduce_op == OpType.MEAN:
            self.dtype = np.float64
        else:
            self.dtype = input.dtype if hasattr(input, "dtype") else None


class SelectListsNode(OpNode):
    """Represents selecting entire lists based on a mask"""

    def __init__(self, input: OpNode, mask: OpNode):
        super().__init__(op_type=OpType.SELECT_LISTS)
        self.input = input
        self.mask = mask
        self.dtype = input.dtype if hasattr(input, "dtype") else None


class CombinationsNode(OpNode):
    """Represents generating combinations from lists"""

    def __init__(
        self,
        input: OpNode,
        n: int,
        replacement: bool = False,
        axis: int = 1,
        fields: list | None = None,
    ):
        super().__init__(op_type=OpType.COMBINATIONS)
        self.input = input
        self.n = n
        self.replacement = replacement
        self.axis = axis
        self.fields = fields
        # Combinations produce record arrays, so dtype is complex
        self.dtype = None


class GetItemNode(OpNode):
    """Represents item access: a field name, index, slice, or a (lazy or
    eager) array key — the latter gives ``lazy_arr[lazy_arr > 5]`` the same
    semantics as eager ``arr[arr > 5]``."""

    def __init__(self, input: OpNode, key):
        super().__init__(op_type=OpType.GETITEM)
        self.input = input
        self.key = key
        # Try to infer dtype from input if it's a known structure
        self.dtype = None
