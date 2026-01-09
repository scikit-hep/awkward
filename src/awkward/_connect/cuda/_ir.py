# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import weakref
from enum import Enum
from typing import Callable

import cupy as cp
import numpy as np

import awkward as ak

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

    # Comparison operations
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    EQ = "eq"
    NE = "ne"

    # List operations
    FILTER = "filter"
    MAP = "map"
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


class IRNode:
    """Base class for IR nodes - Awkward Array compliant"""

    # Class variable to track node IDs
    _id_counter = 0

    def __init__(self, op_type: OpType, dtype=None, shape=None):
        self.op_type = op_type
        self.dtype = dtype
        self.shape = shape
        # Use a class counter instead of id() for more stable hashing
        IRNode._id_counter += 1
        self.node_id = IRNode._id_counter

    def __hash__(self):
        return self.node_id

    def __eq__(self, other):
        """Equality based on node identity, not value"""
        if not isinstance(other, IRNode):
            return False
        return self.node_id == other.node_id

    def __repr__(self):
        return f"{self.__class__.__name__}(op={self.op_type.value}, id={self.node_id})"


class InputNode(IRNode):
    """Represents an input array - stores weak reference to avoid circular refs"""

    def __init__(self, array: ak.Array):
        super().__init__(op_type=OpType.INPUT)
        # Store weak reference to the array to avoid keeping it alive unnecessarily
        # But also keep the layout info we need
        if isinstance(array, ak.Array):
            self._array_ref = weakref.ref(array)
            # Cache the layout type and backend for later use
            self.layout_type = type(array.layout).__name__
            self.backend_name = array.layout._backend.name
            # Store a strong reference only if the array is small
            # For large arrays, we'll just keep the weak ref
            if len(array) < 1000:
                self.array = array
            else:
                self.array = None
        else:
            self._array_ref = None
            self.array = array
            self.layout_type = None
            self.backend_name = None

        self.form = None
        self.buffers = None
        self.metadata = None

    def get_array(self):
        """Get the array, handling weak references properly"""
        if self.array is not None:
            return self.array
        elif self._array_ref is not None:
            arr = self._array_ref()
            if arr is None:
                raise ValueError("Referenced array has been garbage collected")
            return arr
        else:
            raise ValueError("No array reference available")


class ConstantNode(IRNode):
    """Represents a constant value"""

    def __init__(self, value: float | int | np.ndarray | cp.ndarray):
        dtype = None
        if isinstance(value, (int, float)):
            dtype = np.dtype(type(value))
        elif isinstance(value, (np.ndarray, cp.ndarray)):
            dtype = value.dtype
        super().__init__(op_type=OpType.CONSTANT, dtype=dtype)

        # Convert to appropriate numpy/cupy type
        if isinstance(value, (int, float)):
            self.value = value
        elif isinstance(value, np.ndarray):
            self.value = value
        elif isinstance(value, cp.ndarray):
            # Keep as cupy array
            self.value = value
        else:
            # Try to convert to numpy
            self.value = np.asarray(value)


class BinaryOpNode(IRNode):
    """Represents binary operations (add, mul, etc.)"""

    def __init__(self, op_type: OpType, left: IRNode, right: IRNode):
        super().__init__(op_type=op_type)
        self.left = left
        self.right = right

        # Infer dtype from operands if possible
        if left.dtype is not None and right.dtype is not None:
            # Use numpy's type promotion rules
            self.dtype = np.result_type(left.dtype, right.dtype)


class ComparisonNode(IRNode):
    """Represents comparison operations"""

    def __init__(self, op_type: OpType, left: IRNode, right: IRNode):
        super().__init__(op_type=op_type, dtype=np.bool_)
        self.left = left
        self.right = right


class FilterNode(IRNode):
    """Represents filtering operations on lists"""

    def __init__(self, input: IRNode, condition: IRNode):
        super().__init__(op_type=OpType.FILTER)
        self.input = input
        self.condition = condition
        # Output dtype matches input dtype
        self.dtype = input.dtype if hasattr(input, "dtype") else None


class MapNode(IRNode):
    """Represents map operations on lists"""

    def __init__(self, input: IRNode, func: Callable):
        super().__init__(op_type=OpType.MAP)
        self.input = input
        self.func = func


class ReduceNode(IRNode):
    """Represents reduction operations on lists"""

    def __init__(self, input: IRNode, reduce_op: OpType):
        super().__init__(op_type=OpType.REDUCE)
        self.input = input
        self.reduce_op = reduce_op
        # Reductions typically preserve dtype except for MEAN
        if reduce_op == OpType.MEAN:
            self.dtype = np.float64
        else:
            self.dtype = input.dtype if hasattr(input, "dtype") else None


class SelectListsNode(IRNode):
    """Represents selecting entire lists based on a mask"""

    def __init__(self, input: IRNode, mask: IRNode):
        super().__init__(op_type=OpType.SELECT_LISTS)
        self.input = input
        self.mask = mask
        self.dtype = input.dtype if hasattr(input, "dtype") else None


class CombinationsNode(IRNode):
    """Represents generating combinations from lists"""

    def __init__(
        self,
        input: IRNode,
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


class GetItemNode(IRNode):
    """Represents field/index access (e.g., array['field'] or array[0])"""

    def __init__(self, input: IRNode, key: str | int | slice):
        super().__init__(op_type=OpType.GETITEM)
        self.input = input
        self.key = key
        # Try to infer dtype from input if it's a known structure
        self.dtype = None
