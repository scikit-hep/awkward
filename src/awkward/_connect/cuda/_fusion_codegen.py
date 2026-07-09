# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""CUDA lowering for fused element-wise regions (``cuda.compute`` codegen).

This is the backend half of the fusion pass in
``awkward._connect.lazy._fusion``.  Given a :class:`FusedNode` — a run of
element-wise ops, optionally terminated by a reduction — it emits **one**
``cuda.compute`` kernel instead of the per-op launch chain the interpreter
would produce:

* **Element-wise region.**  The leaf columns (flattened list contents) are
  combined with a single ``ZipIterator`` and fed through one
  ``unary_transform`` whose op evaluates the whole sub-expression.  Every
  intermediate stays in registers; nothing is written back to global memory
  between ops.

* **Reduction-terminated region** (``sum`` / ``max`` / ``min``).  The
  element-wise map is wrapped in a ``TransformIterator`` and fed straight into
  ``segmented_reduce``, so the map is fused *into* the reduction kernel — the
  same map-into-reduce idiom already used in ``_compute.py``
  (``_widen_for_reduce`` / ``_nonzero_for_reduce``).  One kernel, no
  materialized transform buffer.

Anything outside the supported shape raises :class:`FusionUnsupported`, and
the executor falls back to the eager (still backend-dispatched, still correct)
evaluation of the fused expression.  This mirrors the migration doc's revert
discipline: fusion is a fast path, never a correctness dependency.

Note: this module is only imported when a fused region's leaves are
CUDA-backed, so importing ``cupy`` / ``cuda.compute`` at call time is safe.
Constants are folded into the compiled op; the op is interned per
(expression, constant, dtype) signature so a stable program compiles each
kernel once (cache-stable, per the plan's hard constraint).
"""

from __future__ import annotations

import functools

import awkward as ak


class FusionUnsupported(Exception):
    """Raised when a fused region cannot be lowered to a single CUDA kernel.

    Signals the executor to fall back to eager evaluation of the region.
    """


# Awkward reduce op -> cuda.compute OpKind name and the identity/init value.
_REDUCE_TO_OPKIND = {
    "sum": ("PLUS", 0),
    "max": ("MAXIMUM", None),  # init supplied from dtype at call time
    "min": ("MINIMUM", None),
}


def _classify_leaf(value):
    """Return ``("column", layout)`` or ``("scalar", value)`` for a leaf.

    Columns are cuda-backed list-of-number arrays (their flat content becomes
    a zipped iterator).  Scalars are numeric constants folded into the op.
    Anything else is unsupported.
    """
    if isinstance(value, ak.Array):
        return "column", value
    # numpy / cupy 0-d or python scalar
    if isinstance(value, (int, float)):
        return "scalar", value
    if hasattr(value, "ndim") and getattr(value, "ndim", None) == 0:
        return "scalar", value.item()
    raise FusionUnsupported(f"unsupported fused leaf of type {type(value).__name__}")


@functools.cache
def _compile_op(source: str, arity: int):
    """Build (and intern) the fused element-wise op from generated source.

    ``source`` is a Python expression over ``t`` (the zipped column tuple) with
    constants already folded in.  Interned on the source string so cuda.compute
    compiles one kernel per distinct fused expression.
    """
    ns: dict = {}
    # ``t`` is the zipped tuple of column values for one element.
    exec(f"def _fused_op(t):\n    return {source}\n", {}, ns)
    return ns["_fused_op"]


def _build_op(node, values):
    """Generate the fused device op plus the ordered list of column layouts.

    Walks the region (via ``node.op_source``) mapping each column leaf to its
    zip slot ``t[k]`` and each scalar leaf to a folded literal.
    """
    columns = []
    leaf_expr = {}
    for leaf_id, value in zip(node.leaf_ids, values, strict=True):
        kind, payload = _classify_leaf(value)
        if kind == "column":
            leaf_expr[leaf_id] = f"t[{len(columns)}]"
            columns.append(payload)
        else:  # scalar constant, folded in
            leaf_expr[leaf_id] = repr(payload)
    if not columns:
        raise FusionUnsupported("fused region has no array leaves")
    source = node.op_source(leaf_expr)
    return _compile_op(source, len(columns)), columns


def _aligned_contents(columns):
    """Return (content_iters, offsets, count) for identically-shaped columns.

    All columns must be list arrays sharing the same offsets so their flat
    contents line up element-for-element.  Otherwise element-wise fusion over
    the flat content is invalid -> unsupported.
    """
    import cupy as cp

    from .helpers import awkward_to_cccl_iterator

    content_iters = []
    ref_offsets = None
    count = None
    for arr in columns:
        it, meta = awkward_to_cccl_iterator(arr)
        offsets = meta["offsets"]
        if offsets is None:
            raise FusionUnsupported("fused column is not a list array")
        if ref_offsets is None:
            ref_offsets, count = offsets, meta["count"]
        elif len(offsets) != len(ref_offsets) or bool((offsets != ref_offsets).any()):
            raise FusionUnsupported("fused columns have mismatched offsets")
        content_iters.append(it)
    return content_iters, ref_offsets, count, cp


def execute_fused_cuda(node, values, reduce_ops):
    """Lower and run a :class:`FusedNode` as a single ``cuda.compute`` kernel.

    ``values`` are the realized leaf arrays (slot order); ``reduce_ops`` is the
    executor's eager reduction table, used only for the unsupported-reduction
    fallback path.  Returns the region result as an Awkward array.
    """
    from cuda.compute import OpKind, TransformIterator, ZipIterator, unary_transform

    from ._compute import segmented_reduce

    op, columns = _build_op(node, values)
    content_iters, offsets, count, cp = _aligned_contents(columns)

    import numpy as np

    zipped = ZipIterator(*content_iters) if len(content_iters) > 1 else content_iters[0]

    if node.reduce_op is None:
        # ---- element-wise: one transform kernel over the zipped columns ----
        out_dtype = np.dtype(node.dtype) if node.dtype is not None else np.float64
        out_content = cp.empty(int(count), dtype=out_dtype)
        unary_transform(d_in=zipped, d_out=out_content, op=op, num_items=int(count))
        template = ak.contents.ListOffsetArray(
            ak.index.Index64(cp.asarray(offsets, dtype=cp.int64)),
            ak.contents.NumpyArray(out_content),
        )
        return ak.Array(template)

    # ---- reduction-terminated: fuse the map into a segmented_reduce --------
    reduce_name = node.reduce_op.value
    if reduce_name not in _REDUCE_TO_OPKIND:
        # e.g. MEAN (two reductions): let the executor apply it eagerly.
        raise FusionUnsupported(f"reduction {reduce_name!r} not fusible")

    opkind_name, init_value = _REDUCE_TO_OPKIND[reduce_name]
    num_segments = len(offsets) - 1
    out_dtype = np.dtype(node.dtype) if node.dtype is not None else np.float64

    # The element-wise map rides into the reduce kernel as a TransformIterator.
    mapped = TransformIterator(zipped, op)
    out = cp.empty(int(num_segments), dtype=out_dtype)
    start_offsets = offsets[:-1]
    end_offsets = offsets[1:]

    if init_value is None:
        info = np.finfo(out_dtype) if out_dtype.kind == "f" else np.iinfo(out_dtype)
        init_value = info.min if opkind_name == "MAXIMUM" else info.max

    segmented_reduce(
        d_in=mapped,
        d_out=out,
        num_segments=int(num_segments),
        start_offsets_in=start_offsets,
        end_offsets_in=end_offsets,
        op=getattr(OpKind, opkind_name),
        init_value=np.asarray([init_value], dtype=out_dtype),
    )
    return ak.Array(out)
