# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import cupy as cp
import numpy as np

from ._segment_algorithms import select_segments
from .helpers import (
    awkward_to_cccl_iterator,
    empty_like,
    reconstruct_with_offsets,
    segment_sizes,
    segmented_select,
    transform_segments,
)

"""
Internal lazy execution utilities for CUDA-backed Awkward IR nodes.
This module provides:
- demand-driven execution
- memoization
- explicit dependency traversal
It is intentionally minimal and internal-only.
"""


class IRNode:
    """
    Base class for lazy IR nodes.
    Subclasses must implement `lower(*inputs)`.
    """

    __slots__ = ("_value", "inputs")

    def __init__(self, *inputs):
        self.inputs = inputs
        self._value = None

    def compute(self):
        """
        Compute this node and return its value.
        Execution is:
        - demand-driven
        - memoized
        - iterative (stack-safe)
        """
        if self._value is not None:
            return self._value

        # Execute in dependency order to prevent recursion depth issues
        for node in topological_order(self):
            if node._value is None:
                # Fetch inputs. Because of topo sort, dependencies are ready.
                args = [
                    inp._value if isinstance(inp, IRNode) else inp
                    for inp in node.inputs
                ]
                node._value = node.lower(*args)

        return self._value

    def lower(self, *args):
        """
        Lower this node to concrete execution.
        Must be implemented by subclasses.
        """
        raise NotImplementedError


# ----------------------------------------------------------------------
# Execution helpers
# ----------------------------------------------------------------------


def compute(node):
    """
    Compute a node or return the value if already concrete.
    """
    if isinstance(node, IRNode):
        return node.compute()
    return node


def reset_cache(node):
    """
    Clear cached values in a graph rooted at `node`.
    """
    visited = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if not isinstance(current, IRNode):
            continue
        if id(current) in visited:
            continue
        visited.add(id(current))

        current._value = None

        for inp in current.inputs:
            if isinstance(inp, IRNode):
                stack.append(inp)


def walk(node):
    """
    Yield all IRNodes reachable from `node` (DFS).
    """
    visited = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if not isinstance(current, IRNode):
            continue
        if id(current) in visited:
            continue
        visited.add(id(current))

        yield current

        for inp in current.inputs:
            if isinstance(inp, IRNode):
                stack.append(inp)


def topological_order(node):
    """
    Return nodes in dependency order (inputs first).
    Iterative implementation (DFS post-order) to ensure stack safety.
    """
    visited = set()
    order = []
    # Stack items: (node, processed_children_flag)
    stack = [(node, False)]

    while stack:
        curr, processed = stack.pop()

        if not isinstance(curr, IRNode):
            continue

        # Optimization: If already visited and added to order, skip.
        # Note: We track visited upon *completion* (adding to order).
        if id(curr) in visited:
            continue

        if processed:
            visited.add(id(curr))
            order.append(curr)
        else:
            # Re-push self with processed=True
            stack.append((curr, True))
            # Push children. Reverse to maintain left-to-right processing order
            # though strictly for topo sort it doesn't matter.
            for inp in reversed(curr.inputs):
                if isinstance(inp, IRNode) and id(inp) not in visited:
                    stack.append((inp, False))

    return order


class Input(IRNode):
    def __init__(self, array):
        super().__init__()
        self.array = array

    def compute(self):
        return self.array

    def lower(self, *args):
        return self.array


class EmptyLike(IRNode):
    def lower(self, array):
        return empty_like(array)


class ToIterator(IRNode):
    def lower(self, array):
        return awkward_to_cccl_iterator(array)


class Filter(IRNode):
    def __init__(self, array, predicate):
        super().__init__(array)
        self.predicate = predicate

    def lower(self, array):
        data_in, meta = awkward_to_cccl_iterator(array)
        offsets_in = meta["offsets"]
        num_items = meta["count"]

        out = empty_like(array)
        data_out, meta_out = awkward_to_cccl_iterator(out)
        offsets_out = meta_out["offsets"]

        segmented_select(
            data_in,
            offsets_in,
            data_out,
            offsets_out,
            self.predicate,
            num_items,
        )

        return reconstruct_with_offsets(out, offsets_out)


class SelectLists(IRNode):
    def __init__(self, array, mask):
        super().__init__(array, mask)

    def lower(self, array, mask):
        data_in, meta = awkward_to_cccl_iterator(array)
        offsets_in = meta["offsets"]

        num_lists = meta["length"]
        num_elements = meta["count"]

        out = empty_like(array)
        data_out, meta_out = awkward_to_cccl_iterator(out)
        offsets_out = meta_out["offsets"]

        d_counts = cp.empty(2, np.int32)

        select_segments(
            data_in,
            offsets_in,
            mask,
            data_out,
            offsets_out,
            d_counts,
            num_elements,
            num_lists,
        )

        _, num_lists_kept = d_counts
        offsets_out = offsets_out[: num_lists_kept + 1]

        return reconstruct_with_offsets(out, offsets_out)


class ListSizes(IRNode):
    def lower(self, array):
        _, meta = awkward_to_cccl_iterator(array)
        return segment_sizes(meta["offsets"])


class TransformLists(IRNode):
    def __init__(self, array, list_size, op):
        super().__init__(array, list_size)
        self.op = op

    def lower(self, array, list_size):
        out = empty_like(array)
        data_in, meta = awkward_to_cccl_iterator(array)
        data_out, _ = awkward_to_cccl_iterator(out)

        transform_segments(
            data_in,
            data_out,
            list_size,
            self.op,
            meta["length"],
        )

        return out
