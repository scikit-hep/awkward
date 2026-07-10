# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Backend-neutral lazy execution core.

This module provides:
- demand-driven execution
- memoization
- explicit dependency traversal

It is intentionally minimal and internal-only.  Backend-specific IR nodes
(``awkward._connect.cuda.ir_nodes``, ``awkward._connect.cpu.ir_nodes``)
subclass ``IRNode`` and implement ``lower()``.
"""

from __future__ import annotations


class IRNode:
    """
    Base class for lazy IR nodes.
    Subclasses must implement ``lower(*inputs)``.
    """

    __slots__ = ("_computed", "_value", "inputs")

    def __init__(self, *inputs):
        self.inputs = inputs
        self._value = None
        self._computed = False

    def compute(self):
        """Compute this node and return its value.

        Execution is demand-driven, memoized, and iterative (stack-safe).

        Returns the node's computed value.
        """
        if self._computed:
            return self._value

        # Execute in dependency order to prevent recursion depth issues
        for node in topological_order(self):
            if not node._computed:
                # Fetch inputs. Because of topo sort, dependencies are ready.
                args = [
                    inp._value if isinstance(inp, IRNode) else inp
                    for inp in node.inputs
                ]
                node._value = node.lower(*args)
                node._computed = True

        return self._value

    def lower(self, *args):
        """Lower this node to concrete execution (implemented by subclasses).

        Args:
            *args: The already-computed values of this node's inputs.

        Returns the node's computed value.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError


# ----------------------------------------------------------------------
# Execution helpers
# ----------------------------------------------------------------------


def compute(node):
    """
    Args:
        node (IRNode or Any): A node to compute, or an already-concrete value.

    Returns ``node.compute()`` for an ``IRNode``, otherwise ``node`` unchanged.
    """
    if isinstance(node, IRNode):
        return node.compute()
    return node


def reset_cache(node):
    """Clear cached values in the graph rooted at ``node``.

    Args:
        node (IRNode): Root of the graph to reset.
    """
    for current in walk(node):
        current._value = None
        current._computed = False


def walk(node):
    """
    Args:
        node (IRNode): Root of the graph to traverse.

    Yields:
        IRNode: Every ``IRNode`` reachable from ``node`` (depth-first).
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
    """Order the graph's nodes so inputs precede the nodes that consume them.

    Iterative (DFS post-order) implementation to ensure stack safety.

    Args:
        node (IRNode): Root of the graph.

    Returns a list of ``IRNode`` in dependency order (inputs first).
    """
    visited = set()
    order = []
    # Stack items: (node, processed_children_flag)
    stack = [(node, False)]

    while stack:
        curr, processed = stack.pop()

        if not isinstance(curr, IRNode):
            continue

        # We track visited upon *completion* (adding to order); duplicate
        # stack entries are skipped here.
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
    """A concrete array used as a leaf of the graph (strong reference).

    Args:
        array: The concrete value this leaf returns when computed.
    """

    __slots__ = ("array",)

    def __init__(self, array):
        super().__init__()
        self.array = array

    def compute(self):
        return self.array

    def lower(self):
        return self.array
