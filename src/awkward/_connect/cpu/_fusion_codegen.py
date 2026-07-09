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

Supported shape: element-wise regions whose array leaves are list arrays with
``NumpyArray`` content sharing identical offsets, plus folded scalar
constants; optionally terminated by ``sum`` / ``mean`` (computed with a
vectorized cumsum so empty sublists are handled).  Anything else raises
:class:`FusionUnsupported` and the executor falls back to eager evaluation.
"""

from __future__ import annotations

import functools

import numpy as np

import awkward as ak

from .helpers import _listoffset_parts


class FusionUnsupported(Exception):
    """Raised when a fused region cannot be lowered to a flat NumPy pass."""


def _classify_leaf(value):
    """Return ``("column", array)`` or ``("scalar", value)`` for a leaf."""
    if isinstance(value, ak.Array):
        return "column", value
    if isinstance(value, (int, float)):
        return "scalar", value
    if hasattr(value, "ndim") and getattr(value, "ndim", None) == 0:
        return "scalar", value.item()
    raise FusionUnsupported(f"unsupported fused leaf of type {type(value).__name__}")


@functools.cache
def _compile_op(source: str):
    """Compile (and intern) the fused expression over the column tuple ``c``.

    Interned on the generated source so a stable program builds each fused
    callable once — the CPU echo of the cuda.compute op-cache requirement.
    """
    ns: dict = {}
    exec(f"def _fused(c):\n    return {source}\n", {"np": np}, ns)
    return ns["_fused"]


def _aligned_columns(columns):
    """Return (contents, offsets) for identically-shaped list columns."""
    contents = []
    ref_offsets = None
    for arr in columns:
        _layout, offsets, content = _listoffset_parts(arr)
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


def _segment_sum(flat, offsets):
    """Vectorized per-sublist sum that tolerates empty sublists."""
    csum = np.zeros(len(flat) + 1, dtype=np.result_type(flat.dtype, np.int64))
    np.cumsum(flat, out=csum[1:])
    return csum[offsets[1:]] - csum[offsets[:-1]]


def execute_fused_cpu(node, values):
    """Lower and run a :class:`FusedNode` as a single flat-buffer NumPy pass.

    Returns the region result as an Awkward array (element-wise) or a NumPy
    array (reduction).  Raises :class:`FusionUnsupported` for shapes outside
    the supported set.
    """
    columns = []
    leaf_expr = {}
    for leaf_id, value in zip(node.leaf_ids, values, strict=True):
        kind, payload = _classify_leaf(value)
        if kind == "column":
            leaf_expr[leaf_id] = f"c[{len(columns)}]"
            columns.append(payload)
        else:
            leaf_expr[leaf_id] = repr(payload)
    if not columns:
        raise FusionUnsupported("fused region has no array leaves")

    contents, offsets = _aligned_columns(columns)
    op = _compile_op(node.op_source(leaf_expr))
    flat_out = op(tuple(contents))  # one pass over the flat content

    if node.reduce_op is None:
        return ak.Array(
            ak.contents.ListOffsetArray(
                ak.index.Index64(offsets.astype(np.int64, copy=False)),
                ak.contents.NumpyArray(np.asarray(flat_out)),
            )
        )

    name = node.reduce_op.value
    if name == "sum":
        return _segment_sum(np.asarray(flat_out), offsets)
    if name == "mean":
        counts = offsets[1:] - offsets[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            return _segment_sum(np.asarray(flat_out), offsets) / counts
    # max/min: no cheap empty-safe vectorization here -> let the executor try
    # the eager reducer instead.
    raise FusionUnsupported(f"reduction {name!r} not lowered on CPU")
