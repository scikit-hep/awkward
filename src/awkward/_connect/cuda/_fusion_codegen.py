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

* **Reduction-terminated region** (``sum`` only).  The element-wise map is
  wrapped in a ``TransformIterator`` and fed straight into ``segmented_reduce``,
  so the map is fused *into* the reduction kernel — the same map-into-reduce
  idiom already used in ``_compute.py`` (``_widen_for_reduce`` /
  ``_nonzero_for_reduce``).  One kernel, no materialized transform buffer.
  ``max`` / ``min`` (empty sublists reduce to a missing value under eager
  Awkward) and ``mean`` (two reductions) are *not* fused — they raise
  :class:`FusionUnsupported` and the executor applies the eager reducer.

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

from awkward._connect.lazy._fusion import py_scalar_literal
from awkward._connect.lazy._layout import is_fusible_numeric_list


class FusionUnsupported(Exception):
    """Raised when a fused region cannot be lowered to a single CUDA kernel.

    Signals the executor to fall back to eager evaluation of the region.
    """


# Reductions fused on the device.  Only ``sum`` is fused: its CCCL identity
# (0) matches ``ak.sum(..., axis=-1)`` empty-list semantics exactly.  ``max`` /
# ``min`` reduce empty sublists to the *missing value* (option/None) under
# eager Awkward, which a plain ``segmented_reduce`` seeded with the dtype
# extreme cannot reproduce; ``mean`` is two reductions.  Those fall back to the
# eager reducer (``FusionUnsupported``) so results stay identical.
_FUSIBLE_REDUCE = {"sum": "PLUS"}


def _sum_accumulator_dtype(dtype):
    """Widen small integer/bool dtypes like NumPy's ``sum`` accumulator.

    Mirrors ``ak.sum`` so the fused sum's dtype matches the eager result
    (int8/16/32 -> int64, unsigned -> uint64, bool -> int64; floats unchanged).
    """
    import numpy as np

    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return np.dtype(np.int64)
    if dtype.kind == "i" and dtype.itemsize < 8:
        return np.dtype(np.int64)
    if dtype.kind == "u" and dtype.itemsize < 8:
        return np.dtype(np.uint64)
    return dtype


def _infer_out_dtype(op, columns, single):
    """Dtype of the fused element-wise result, matching NumPy/eager promotion.

    Runs the folded op on a one-element sample of each column's leaf dtype (so
    folded scalar constants and true type promotion are honoured) instead of
    defaulting to ``float64``.
    """
    import numpy as np

    from awkward._connect.lazy._ir import _leaf_numpy_dtype

    dtypes = [_leaf_numpy_dtype(col) or np.dtype(np.float64) for col in columns]
    try:
        samples = [np.ones(1, dtype=d) for d in dtypes]
        probe = samples[0] if single else tuple(samples)
        with np.errstate(all="ignore"):  # divide-by-zero etc. in the probe only
            return np.asarray(op(probe)).dtype
    except Exception:  # noqa: BLE001
        # A value-domain error in the probe (e.g. int ** negative int) need not
        # reflect the real data; fall back to NumPy input promotion.
        return np.result_type(*dtypes)


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
    element accessor and each scalar leaf to a folded literal.  With one column
    the transform element is the scalar itself (``t``); with several it is a
    zipped tuple (``t[k]``).
    """
    col_slot = {}
    scalar_expr = {}
    columns = []
    for leaf_id, value in zip(node.leaf_ids, values, strict=True):
        kind, payload = _classify_leaf(value)
        if kind == "column":
            col_slot[leaf_id] = len(columns)
            columns.append(payload)
        else:  # scalar constant, folded in
            scalar_expr[leaf_id] = py_scalar_literal(payload)
    if not columns:
        raise FusionUnsupported("fused region has no array leaves")

    single = len(columns) == 1
    leaf_expr = dict(scalar_expr)
    for leaf_id, slot in col_slot.items():
        leaf_expr[leaf_id] = "t" if single else f"t[{slot}]"
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
        if not is_fusible_numeric_list(arr):
            raise FusionUnsupported(
                "column is not a plain numeric var-list (strings, regular, "
                "indexed, parametered, or record layouts fall back to eager)"
            )
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


def execute_fused_cuda(node, values):
    """Lower and run a :class:`FusedNode` as a single ``cuda.compute`` kernel.

    ``values`` are the realized leaf arrays (slot order).  Returns the region
    result as an Awkward array, or raises :class:`FusionUnsupported` (e.g. a
    non-``sum`` reduction) so the executor applies the eager reducer instead.
    """
    from cuda.compute import OpKind, TransformIterator, ZipIterator, unary_transform

    from ._compute import make_segment_views, segmented_reduce

    op, columns = _build_op(node, values)
    content_iters, offsets, count, cp = _aligned_contents(columns)

    import numpy as np

    single = len(content_iters) == 1
    zipped = content_iters[0] if single else ZipIterator(*content_iters)
    # Output dtype from the actual leaf dtypes (matches NumPy/eager promotion),
    # never a blanket float64 that would turn an integer input floating.
    map_dtype = _infer_out_dtype(op, columns, single)

    if node.reduce_op is None:
        # ---- element-wise: one transform kernel over the zipped columns ----
        out_content = cp.empty(int(count), dtype=map_dtype)
        unary_transform(d_in=zipped, d_out=out_content, op=op, num_items=int(count))
        template = ak.contents.ListOffsetArray(
            ak.index.Index64(cp.asarray(offsets, dtype=cp.int64)),
            ak.contents.NumpyArray(out_content),
        )
        return ak.Array(template)

    # ---- reduction-terminated: fuse the map into a segmented_reduce --------
    reduce_name = node.reduce_op.value
    if reduce_name not in _FUSIBLE_REDUCE:
        # max/min (empty -> missing value) and mean (two reductions) can't match
        # eager semantics from a single seeded segmented_reduce; fall back.
        raise FusionUnsupported(f"reduction {reduce_name!r} not fused on device")

    num_segments = len(offsets) - 1
    out_dtype = _sum_accumulator_dtype(map_dtype)  # widen small ints like ak.sum

    # The element-wise map rides into the reduce kernel as a TransformIterator,
    # so the map + segmented reduction are one kernel with no intermediate
    # buffer -- the same map-into-reduce idiom as _compute.py's reducers.
    mapped = TransformIterator(zipped, op)
    out = cp.empty(int(num_segments), dtype=out_dtype)
    start_offsets, end_offsets = make_segment_views(offsets)

    segmented_reduce(
        d_in=mapped,
        d_out=out,
        num_segments=int(num_segments),
        start_offsets_in=start_offsets,
        end_offsets_in=end_offsets,
        op=getattr(OpKind, _FUSIBLE_REDUCE[reduce_name]),
        h_init=np.asarray(0, dtype=out_dtype),  # sum identity; empty -> 0 (== ak.sum)
    )
    return ak.Array(out)
