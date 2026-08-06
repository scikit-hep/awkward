# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""CPU lowering for fused element-wise regions (the NumPy counterpart of
``awkward._connect.cuda._fusion_codegen``).

The interpreter runs every IR node as a separate eager Awkward op, so an
N-op element-wise chain over a jagged array pays the full ``ak.Array``
dispatch/broadcast cost N times and materializes N intermediate arrays.  This
module instead evaluates a whole :class:`FusedNode` region **once, directly on
the flat NumPy content buffer** — the CPU analogue of collapsing the chain
into one kernel: the per-op Awkward dispatch is paid zero times (only raw
NumPy ufuncs run) and only the single final result is rewrapped with the
shared offsets.

Supported shape: element-wise regions whose array leaves are parameter-free
``ListOffsetArray``/``ListArray`` layouts with ``NumpyArray`` content sharing
identical offsets, plus folded scalar constants.  Parameterized layouts
(strings, behavior-carrying lists) and ``RegularArray`` are rejected so the
eager fallback keeps their special semantics and types.  This lowers the
**element-wise map only** — a terminating reduction is applied by the
executor via the eager Awkward reducer on the fused result, so
``compute(fuse=True)`` and ``compute(fuse=False)`` return the identical
Awkward-typed reduction (same masking / empty-list / backend semantics).
Anything else raises :class:`FusionUnsupported` and the executor falls back to
eager evaluation.
"""

from __future__ import annotations

import functools

import numpy as np

import awkward as ak
from awkward._connect.lazy._fusion import py_scalar_literal
from awkward._connect.lazy._layout import is_fusible_numeric_list


class FusionUnsupported(Exception):
    """Raised when a fused region cannot be lowered to a flat NumPy pass."""


def _classify_leaf(value):
    """
    Args:
        value: A realized leaf value (an ``ak.Array`` column or a scalar).

    Returns ``("column", array)`` for an ``ak.Array`` leaf or
    ``("scalar", value)`` for a numeric scalar.

    Raises:
        FusionUnsupported: If ``value`` is neither an array nor a scalar.
    """
    if isinstance(value, ak.Array):
        return "column", value
    if isinstance(value, (int, float)):
        return "scalar", value
    if hasattr(value, "ndim") and getattr(value, "ndim", None) == 0:
        return "scalar", value.item()
    raise FusionUnsupported(f"unsupported fused leaf of type {type(value).__name__}")


@functools.lru_cache(maxsize=128)
def _compile_op(source: str):
    """Compile (and intern) the fused expression over the column tuple ``c``.

    Interned on the generated source so a stable program builds each fused
    callable once — the CPU echo of the cuda.compute op-cache requirement.
    Bounded: constants are folded into the source, so a loop generating distinct
    constants must not grow the cache without limit.

    Args:
        source (str): Python expression over the column tuple ``c`` with
            constants folded in.

    Returns a callable ``_fused(c)`` evaluating the expression.
    """
    ns: dict = {}
    exec(f"def _fused(c):\n    return {source}\n", {"np": np}, ns)
    return ns["_fused"]


def _validated_parts(arr):
    """Extract the flat parts of a fusible list column.

    Only the shape :func:`is_fusible_numeric_list` accepts qualifies; everything
    else — ``RegularArray`` (the fallback preserves the regular type),
    parameterized layouts (strings and other behaviors have non-element-wise
    semantics), option types, records, flat arrays — falls back to eager.

    Args:
        arr (ak.Array): Candidate column.

    Returns an ``(offsets, content)`` tuple of NumPy arrays: the list offsets
    and the flattened numeric content.

    Raises:
        FusionUnsupported: If ``arr`` is not a plain numeric var-list.
    """
    if not is_fusible_numeric_list(arr):
        raise FusionUnsupported(
            "column is not a plain numeric var-list (strings, regular, "
            "indexed, parametered, or record layouts fall back to eager)"
        )
    layout = arr.layout.to_ListOffsetArray64(True)
    offsets = np.asarray(layout.offsets.data)
    content = np.asarray(layout.content.data)[: offsets[-1]]
    return offsets, content


def _aligned_columns(columns):
    """
    Args:
        columns (list): The ``ak.Array`` column leaves of the region.

    Returns a ``(contents, offsets)`` tuple: the flattened numeric contents of
    every column and the shared offsets buffer.

    Raises:
        FusionUnsupported: If the columns have mismatched offsets, or the region
            has no array leaves.
    """
    contents = []
    ref_offsets = None
    for arr in columns:
        offsets, content = _validated_parts(arr)
        if ref_offsets is None:
            ref_offsets = offsets
        elif len(offsets) != len(ref_offsets) or not np.array_equal(
            offsets, ref_offsets
        ):
            raise FusionUnsupported("fused columns have mismatched offsets")
        contents.append(content)
    if ref_offsets is None:
        raise FusionUnsupported("fused region has no array leaves")
    return contents, ref_offsets


def execute_fused_cpu(node, values):
    """Lower a :class:`FusedNode`'s element-wise map to one flat-buffer pass.

    The reduction (if ``node.reduce_op`` is set) is **not** applied here — the
    executor applies the eager Awkward reducer to this result, so the fused and
    interpreter paths return the identical Awkward-typed reduction.

    Args:
        node (FusedNode): The fused region to lower.
        values (list): Realized leaf values, in ``node.leaves`` order.

    Returns the fused element-wise result as an Awkward list array.

    Raises:
        FusionUnsupported: For shapes outside the supported set (non-numeric
            layouts, mismatched offsets, or no array leaves).
    """
    columns = []
    leaf_expr = {}
    for leaf_id, value in zip(node.leaf_ids, values, strict=True):
        kind, payload = _classify_leaf(value)
        if kind == "column":
            leaf_expr[leaf_id] = f"c[{len(columns)}]"
            columns.append(payload)
        else:
            leaf_expr[leaf_id] = py_scalar_literal(payload)
    if not columns:
        raise FusionUnsupported("fused region has no array leaves")

    contents, offsets = _aligned_columns(columns)
    try:
        op = _compile_op(node.op_source(leaf_expr))
    except SyntaxError as exc:
        # e.g. a region deep enough to exceed the parser's nesting limit
        raise FusionUnsupported(f"cannot compile fused source: {exc}") from exc
    flat_out = op(tuple(contents))  # one pass over the flat content

    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(offsets.astype(np.int64, copy=False)),
            ak.contents.NumpyArray(np.asarray(flat_out)),
        )
    )
