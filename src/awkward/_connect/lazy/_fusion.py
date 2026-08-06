# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Kernel-fusion pass for the operator-level IR (backend-neutral).

The operator IR built by ``LazyAwkwardArray`` (see ``_ir.py``) is a directed
acyclic *expression graph*: it captures the user's intent ("scale these,
filter, then take the mean") without executing anything.  The default
``IRExecutor`` is a pure interpreter — it walks that graph and dispatches
every node to an eager Awkward op, materializing one intermediate array per
node.  On a GPU that means one kernel launch (and one round-trip through
global memory) per operation.

This module implements **Dynamic Compilation**: a pass that rewrites the
expression graph, collapsing each maximal run of element-wise operations
(optionally terminated by a reduction) into a single :class:`FusedNode`.  A
``FusedNode`` carries

* ``leaves`` — the boundary inputs that must be materialized first, and
* ``expr``   — one *cache-stable* callable that evaluates the whole
  element-wise sub-expression from those leaves in a single pass.

A backend then realizes a ``FusedNode`` as **one** kernel: the CUDA backend
(``awkward._connect.cuda._fusion_codegen``) lowers it to a single
``cuda.compute`` transform over a ``ZipIterator`` of the leaf columns, so
intermediates stay in registers / L1 instead of being written back to global
memory between ops.  Any backend without a fusion codegen simply evaluates
``expr`` eagerly — still a win at the IR level (one dispatch, no per-node
memo traffic), and always numerically identical to the interpreter.

Fusion policy — *single-use fusion*.  A node is absorbed into its parent
region only if it is element-wise **and** has exactly one consumer in the
graph.  A node with fan-out > 1 becomes a region boundary (a shared leaf),
so shared sub-expressions are materialized once rather than recomputed inside
every consumer.  This keeps the memoization guarantee of the interpreter
while still fusing linear chains, and it is safe: boundaries are exactly the
points the interpreter would have materialized anyway.

Everything here is iterative (explicit stacks, no recursion): expression
graphs are user-sized — a loop appending ``expr = expr + x`` thousands of
times must fuse and evaluate without hitting the Python recursion limit.
"""

from __future__ import annotations

import math
import operator
from collections import defaultdict

import numpy as np

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

# ---------------------------------------------------------------------------
# Op tables
# ---------------------------------------------------------------------------

#: Element-wise ops eligible for fusion.  Their operands are read from
#: ``.left`` / ``.right`` and each maps 1:1 to a Python operator that works
#: element-wise on realized (NumPy/CuPy-backed) Awkward arrays.
ELEMENTWISE_OPS = {
    OpType.ADD: operator.add,
    OpType.SUB: operator.sub,
    OpType.MUL: operator.mul,
    OpType.DIV: operator.truediv,
    OpType.POW: operator.pow,
    OpType.MOD: operator.mod,
    OpType.FLOORDIV: operator.floordiv,
    OpType.LT: operator.lt,
    OpType.LE: operator.le,
    OpType.GT: operator.gt,
    OpType.GE: operator.ge,
    OpType.EQ: operator.eq,
    OpType.NE: operator.ne,
}

#: Symbols used for fused-expression rendering (both the human-readable
#: ``expr_text`` and the backend codegen source).
_OP_SYMBOL = {
    OpType.ADD: "+",
    OpType.SUB: "-",
    OpType.MUL: "*",
    OpType.DIV: "/",
    OpType.POW: "**",
    OpType.MOD: "%",
    OpType.FLOORDIV: "//",
    OpType.LT: "<",
    OpType.LE: "<=",
    OpType.GT: ">",
    OpType.GE: ">=",
    OpType.EQ: "==",
    OpType.NE: "!=",
}


def is_elementwise(node: OpNode) -> bool:
    """
    Args:
        node (OpNode): Node to test.

    Returns True for a binary/comparison node whose op is in
    :data:`ELEMENTWISE_OPS` and can therefore be fused.
    """
    return (
        isinstance(node, (BinaryOpNode, ComparisonNode))
        and node.op_type in ELEMENTWISE_OPS
    )


def _children(node: OpNode) -> tuple[OpNode, ...]:
    """
    Args:
        node (OpNode): Node whose operands to return.

    Returns the tuple of ``OpNode`` operands of ``node`` (empty for leaves such
    as ``InputNode`` / ``ConstantNode``).
    """
    if isinstance(node, (BinaryOpNode, ComparisonNode)):
        return (node.left, node.right)
    if isinstance(node, FilterNode):
        return (node.input, node.condition)
    if isinstance(node, SelectListsNode):
        return (node.input, node.mask)
    if isinstance(node, (ReduceNode, CombinationsNode)):
        return (node.input,)
    if isinstance(node, GetItemNode):
        if isinstance(node.key, OpNode):
            return (node.input, node.key)
        return (node.input,)
    if isinstance(node, FusedNode):
        return tuple(node.leaves)
    return ()  # InputNode, ConstantNode


def _walk(root: OpNode):
    """
    Args:
        root (OpNode): Node to start the traversal from.

    Yields:
        OpNode: Every distinct node reachable from ``root`` (depth-first).
    """
    seen: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        yield node
        stack.extend(_children(node))


def topo_order(root: OpNode) -> list[OpNode]:
    """
    Args:
        root (OpNode): Root of the expression graph.

    Returns the graph's nodes in dependency order (children before parents),
    computed iteratively so deep graphs do not hit the recursion limit.
    """
    seen: set[int] = set()
    order: list[OpNode] = []
    stack: list[tuple[OpNode, bool]] = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            order.append(node)
            continue
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        stack.append((node, True))
        for child in reversed(_children(node)):
            if child.node_id not in seen:
                stack.append((child, False))
    return order


def consumer_counts(root: OpNode) -> dict[int, int]:
    """
    Args:
        root (OpNode): Root of the expression graph.

    Returns a dict mapping ``node_id`` to the number of parents referencing it
    within the graph. The graph root has count 0; a count > 1 marks a fan-out
    point, which the fusion policy treats as a region boundary (materialize
    once, share).
    """
    counts: dict[int, int] = defaultdict(int)
    for node in _walk(root):
        for child in _children(node):
            counts[child.node_id] += 1
    return counts


# ---------------------------------------------------------------------------
# FusedNode
# ---------------------------------------------------------------------------


def _eval_program(program, values):
    """Evaluate a postorder (RPN) instruction list with an explicit stack.

    Stack-based so arbitrarily deep fused regions evaluate without recursion.

    Args:
        program (tuple): Sequence of ``(fn, slot)`` entries. A leaf load has
            ``fn is None`` and pushes ``values[slot]``; an op has ``fn`` set and
            pops two operands to push ``fn(left, right)``.
        values (list): Realized leaf values indexed by slot.

    Returns the value left on top of the stack (the region's result).
    """
    stack = []
    for fn, slot in program:
        if fn is None:
            stack.append(values[slot])
        else:
            right = stack.pop()
            left = stack.pop()
            stack.append(fn(left, right))
    return stack[-1]


class FusedNode(OpNode):
    """A run of element-wise ops (optionally terminated by a reduction),
    collapsed into a single compiled expression over boundary ``leaves``.

    ``expr`` closes only over values fixed at fusion time (never per-call
    state), so it is cache-stable — a hard requirement for the ``cuda.compute``
    op cache.

    Attributes:
        leaves (list): Boundary input nodes, in slot order, materialized first.
        expr (callable): Takes the realized leaf values (in ``leaves`` order)
            and returns the element-wise result.
        reduce_op (OpType or None): If set, the reduction the executor/backend
            applies as the fused reduction stage.
        root_node (OpNode): The region's element-wise root; backend codegen
            re-reads its op structure.
        leaf_ids (list): Original leaf ``node_id``s in slot order, aligned with
            ``leaves``; codegen maps each to its slot.
        expr_text (str): Human-readable rendering of the fused expression.
    """

    def __init__(
        self, leaves, expr, root_node, leaf_ids=None, reduce_op=None, expr_text=""
    ):
        dtype = getattr(root_node, "dtype", None)
        if reduce_op is OpType.MEAN:
            dtype = np.dtype(np.float64)
        super().__init__(op_type=OpType.FUSED, dtype=dtype)
        self.leaves = list(leaves)
        self.expr = expr
        self.reduce_op = reduce_op
        # Kept for backend codegen (it re-reads the op structure) and debug.
        self.root_node = root_node
        # Original leaf node ids in slot order, aligned with ``self.leaves``.
        # Backend codegen walks ``root_node`` and maps each original leaf id to
        # its slot; the rewritten ``leaves`` supply the slot values.
        self.leaf_ids = list(leaf_ids) if leaf_ids is not None else []
        self.expr_text = expr_text

    def op_source(self, leaf_expr: dict) -> str:
        """Render the region as a Python expression string for codegen.

        Args:
            leaf_expr (dict): Maps each original leaf ``node_id`` to the source
                snippet that produces its value (e.g. ``"t[0]"`` for a zipped
                column, or a numeric literal for a folded constant).

        Returns the fused element-wise expression as Python source.
        """
        return emit_source(self.root_node, leaf_expr)

    def evaluate(self, values):
        """Eager fallback: evaluate the fused element-wise expression.

        Used by any backend without a dedicated fusion codegen. The reduction
        (if any) is applied by the caller so backends can fuse it into the same
        kernel instead.

        Args:
            values (list): Realized leaf values, in ``leaves`` order.

        Returns the element-wise result (before any reduction).
        """
        return self.expr(values)

    def __repr__(self):
        red = f", reduce={self.reduce_op.value}" if self.reduce_op else ""
        return (
            f"FusedNode(id={self.node_id}, leaves={len(self.leaves)}"
            f"{red}, expr={self.expr_text!r})"
        )


# ---------------------------------------------------------------------------
# Region construction
# ---------------------------------------------------------------------------


def _build_region(root: OpNode, counts: dict[int, int]):
    """Collect the fused region rooted at element-wise ``root``.

    Args:
        root (OpNode): Element-wise node at the top of the region.
        counts (dict): Consumer counts from :func:`consumer_counts`, used to
            decide which element-wise children are absorbed vs. become leaves.

    Returns a ``(leaves, expr, expr_text)`` tuple, where ``leaves`` are the
    ordered boundary ``OpNode``s, ``expr`` is a cache-stable callable evaluating
    the region from a list of leaf values, and ``expr_text`` renders it.
    """
    leaves: list[OpNode] = []
    leaf_slot: dict[int, int] = {}

    def is_interior(node: OpNode) -> bool:
        # The root is always interior; other element-wise nodes are absorbed
        # only when single-use (fan-out == 1 -> unique consumer is this parent).
        if node is root:
            return True
        return is_elementwise(node) and counts.get(node.node_id, 0) == 1

    # Pre-order DFS: assign leaf slots in first-encounter (left-to-right) order.
    stack = [root]
    while stack:
        node = stack.pop()
        if is_interior(node):
            left, right = node.left, node.right
            stack.append(right)
            stack.append(left)
        elif node.node_id not in leaf_slot:
            leaf_slot[node.node_id] = len(leaves)
            leaves.append(node)

    # Postorder (RPN) program over the region: leaf loads and binary ops.
    program: list[tuple] = []
    post: list[tuple[OpNode, bool]] = [(root, False)]
    while post:
        node, processed = post.pop()
        if processed:
            program.append((ELEMENTWISE_OPS[node.op_type], None))
            continue
        if node.node_id in leaf_slot:
            program.append((None, leaf_slot[node.node_id]))
            continue
        post.append((node, True))
        post.append((node.right, False))
        post.append((node.left, False))

    def expr(values, _program=tuple(program)):
        return _eval_program(_program, values)

    text = emit_source(root, {nid: f"${slot}" for nid, slot in leaf_slot.items()})
    return leaves, expr, text


def py_scalar_literal(value) -> str:
    """Emit a folded scalar constant as valid Python source.

    ``repr`` renders ``inf`` / ``nan`` as bare names that do not exist in the
    compiled op's namespace (a ``NameError`` at exec time, which used to escape
    the fusion fallback), so this emits explicit ``float(...)`` calls instead.
    (The CUDA codegen instead resolves the bare names via exec globals — numba
    cannot compile ``float(str)`` in device code — so both backends stay correct
    for non-finite constants.)

    Args:
        value: The folded constant (typically an ``int`` or ``float``).

    Returns a string of Python source that evaluates to ``value``.
    """
    if isinstance(value, float):
        if math.isinf(value):
            return "float('inf')" if value > 0 else "float('-inf')"
        if math.isnan(value):
            return "float('nan')"
    return repr(value)


def emit_source(root_node: OpNode, leaf_expr: dict) -> str:
    """Walk an element-wise region and emit an equivalent Python expression.

    Shared with the CUDA codegen so the fused-op structure has a single source
    of truth.

    Args:
        root_node (OpNode): Element-wise root of the region.
        leaf_expr (dict): Maps each original leaf ``node_id`` to its source
            snippet. Interior nodes (not in ``leaf_expr``) must be element-wise;
            each becomes an infix application.

    Returns the region as a fully-parenthesized Python expression string.
    """
    out: list[str] = []
    stack: list[tuple[OpNode, bool]] = [(root_node, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            right = out.pop()
            left = out.pop()
            out.append(f"({left} {_OP_SYMBOL[node.op_type]} {right})")
            continue
        snippet = leaf_expr.get(node.node_id)
        if snippet is not None:
            out.append(snippet)
            continue
        stack.append((node, True))
        stack.append((node.right, False))
        stack.append((node.left, False))
    return out[0]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _rebuild_boundary(node: OpNode, rewritten):
    """Reconstruct a non-fused (boundary) node with rewritten children.

    Args:
        node (OpNode): The boundary node to rebuild.
        rewritten (dict): Maps ``node_id`` to the already-rewritten node;
            children are guaranteed present because :func:`fuse` processes in
            topological order.

    Returns the rebuilt boundary node (input/constant/fused nodes are returned
    unchanged, which keeps :func:`fuse` idempotent).

    Raises:
        NotImplementedError: If ``node`` is of an unhandled type.
    """
    if isinstance(node, (InputNode, ConstantNode, FusedNode)):
        # True leaves and already-fused regions are identity: this makes the
        # pass idempotent (``fuse(fuse(g)) == fuse(g)``) and preserves the
        # executor memo / input references.
        return node
    if isinstance(node, FilterNode):
        return FilterNode(
            rewritten[node.input.node_id], rewritten[node.condition.node_id]
        )
    if isinstance(node, SelectListsNode):
        return SelectListsNode(
            rewritten[node.input.node_id], rewritten[node.mask.node_id]
        )
    if isinstance(node, ReduceNode):
        return ReduceNode(rewritten[node.input.node_id], node.reduce_op)
    if isinstance(node, CombinationsNode):
        return CombinationsNode(
            rewritten[node.input.node_id],
            node.n,
            node.replacement,
            node.axis,
            node.fields,
        )
    if isinstance(node, GetItemNode):
        key = node.key
        if isinstance(key, OpNode):
            key = rewritten[key.node_id]
        return GetItemNode(rewritten[node.input.node_id], key)
    raise NotImplementedError(f"cannot rebuild node of type {type(node).__name__}")


def _fuses_reduce(node: OpNode, counts: dict[int, int]) -> bool:
    """
    Args:
        node (OpNode): Node to test.
        counts (dict): Consumer counts from :func:`consumer_counts`.

    Returns True when ``node`` is a reduction over a single-use element-wise
    input, so that transform+reduce collapses into one ``FusedNode`` (one
    kernel).
    """
    return (
        isinstance(node, ReduceNode)
        and is_elementwise(node.input)
        and counts.get(node.input.node_id, 0) == 1
    )


def fuse(root: OpNode) -> OpNode:
    """Rewrite the expression graph, collapsing element-wise regions.

    Element-wise regions become :class:`FusedNode`; reductions whose
    (single-use) input is element-wise are fused into one ``FusedNode`` carrying
    ``reduce_op`` (transform+reduce in one kernel). Structural nodes (filter,
    select-lists, combinations, getitem) are kept as boundaries, with their
    children rewritten. Idempotent by ``node_id``, so shared sub-graphs are
    fused once and reused.

    Args:
        root (OpNode): Root of the original expression graph.

    Returns the new (fused) root ``OpNode``.
    """
    counts = consumer_counts(root)
    order = topo_order(root)

    # Interior nodes — absorbed into their (unique, absorbing) consumer's
    # region — are skipped: only region roots get a FusedNode of their own.
    absorbed: set[int] = set()
    for node in order:
        if is_elementwise(node) or isinstance(node, ReduceNode):
            for child in _children(node):
                if (
                    is_elementwise(child)
                    and counts.get(child.node_id, 0) == 1
                    and (is_elementwise(node) or child is node.input)
                ):
                    absorbed.add(child.node_id)

    rewritten: dict[int, OpNode] = {}
    for node in order:
        if node.node_id in absorbed:
            continue
        if _fuses_reduce(node, counts):
            leaves, expr, text = _build_region(node.input, counts)
            result: OpNode = FusedNode(
                [rewritten[leaf.node_id] for leaf in leaves],
                expr,
                root_node=node.input,
                leaf_ids=[leaf.node_id for leaf in leaves],
                reduce_op=node.reduce_op,
                expr_text=text,
            )
        elif is_elementwise(node):
            leaves, expr, text = _build_region(node, counts)
            result = FusedNode(
                [rewritten[leaf.node_id] for leaf in leaves],
                expr,
                root_node=node,
                leaf_ids=[leaf.node_id for leaf in leaves],
                expr_text=text,
            )
        else:
            result = _rebuild_boundary(node, rewritten)
        rewritten[node.node_id] = result

    return rewritten[root.node_id]


def fusion_stats(root: OpNode) -> dict[str, int]:
    """Summarize what fusion does to a graph (for tests / introspection).

    Args:
        root (OpNode): Root of the original expression graph.

    Returns a dict with:

    - ``elementwise_before``: fusible nodes in the original graph;
    - ``fused_regions``: ``FusedNode``s produced by the pass;
    - ``materialized``: nodes the executor still evaluates individually after
      fusion.
    """
    before = sum(1 for n in _walk(root) if is_elementwise(n))
    fused_root = fuse(root)
    fused_regions = sum(1 for n in _walk(fused_root) if isinstance(n, FusedNode))
    materialized = sum(1 for _ in _walk(fused_root))
    return {
        "elementwise_before": before,
        "fused_regions": fused_regions,
        "materialized": materialized,
    }
