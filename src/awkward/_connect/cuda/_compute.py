# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import functools

from cuda.compute import (
    CountingIterator,
    DiscardIterator,
    OpKind,
    TransformIterator,
    ZipIterator,
    inclusive_scan,
    reduce_into,
    unary_transform,
)
from cuda.compute import (
    segmented_reduce as _segmented_reduce_raw,
)
from numba import cuda

from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy

cupy_nplike = Cupy.instance()
cp = cupy_nplike._module

numpy_nplike = Numpy.instance()
np = numpy_nplike._module

# NOTE: error messages in this module deliberately keep the legacy
# "... in compiled CUDA code (<kernel name>)" suffix even though these checks
# run host-side in Python: the generated CUDA unit tests (and downstream
# users) match on that exact text, so it must stay in sync with the messages
# raised from compiled kernels.

# Cache for cuda.compute availability
_cuda_compute_available: bool | None = None


def is_available() -> bool:
    global _cuda_compute_available

    if _cuda_compute_available is not None:
        return _cuda_compute_available

    try:
        import cuda.compute  # noqa: F401

        _cuda_compute_available = True
    except ImportError:
        _cuda_compute_available = False

    return _cuda_compute_available


def make_segment_views(offsets):
    """
    Returns (starts, stops) views for segmented operations.
    """
    return offsets[:-1], offsets[1:]


def segmented_reduce(*, max_segment_size=None, **kwargs):
    """Thin wrapper over ``cuda.compute.segmented_reduce`` that supplies the
    ``max_segment_size`` hint (number of elements in the largest segment) when a
    caller doesn't pass one. cuda.compute uses it to dispatch to the
    block/warp/thread implementation, which removes the segmented-reduce
    slowness that originally motivated the ``unary_transform`` workarounds.

    The hint only affects performance, never the result. It's derived from the
    array-backed offset views; if a caller passes iterators (no cheap max), we
    leave it ``None`` and let cuda.compute choose.
    """
    if max_segment_size is None:
        n = kwargs.get("num_segments", 0)
        start = kwargs.get("start_offsets_in")
        stop = kwargs.get("end_offsets_in")
        if n and start is not None and stop is not None:
            try:
                max_segment_size = int((stop[:n] - start[:n]).max())
            except (TypeError, AttributeError, ValueError):
                max_segment_size = None
    return _segmented_reduce_raw(**kwargs, max_segment_size=max_segment_size)


def normalize_index_dtype(dtype):
    dt = cp.dtype(dtype)
    if dt.kind in ("u", "i"):
        return cp.int64
    return dt


def infer_complex_dtype(dtype):
    dt = cp.dtype(dtype)
    if dt == cp.float32:
        return cp.complex64
    if dt == cp.float64:
        return cp.complex128
    raise TypeError(f"Expected float32/float64 interleaved complex buffer, got {dt}")


@functools.cache
def _make_widening_cast(in_type, out_type):
    """Interned scalar cast op for a (in_type -> out_type) widening.

    Returned with a stable identity (cached per dtype pair) so cuda.compute
    builds one kernel per pair instead of recompiling a fresh closure each call.
    The op only needs to widen; cuda.compute uses the return annotation to pick
    the output type, so an `int()` / `float()` body suffices.
    """
    if np.dtype(out_type).kind == "f":

        def _cast(x):
            return float(x)

    else:

        def _cast(x):
            return int(x)

    _cast.__annotations__ = {"x": in_type, "return": out_type}
    return _cast


def _widen_for_reduce(input_data, out_dtype):
    """Fuse a widening cast into a downstream segmented_reduce.

    Returns ``input_data`` unchanged when no cast is needed; otherwise a
    ``TransformIterator`` that casts on the fly, so the reduction never sees (or
    allocates) a materialised ``astype`` copy of the whole input. Equivalent to
    ``input_data.astype(out_dtype, copy=False)`` but without the buffer.
    """
    if input_data.dtype == out_dtype:
        return input_data
    return TransformIterator(
        input_data,
        _make_widening_cast(input_data.dtype.type, np.dtype(out_dtype).type),
    )


@functools.cache
def _make_is_nonzero(in_type, out_type):
    """Interned ``x -> 1 if x != 0 else 0`` map op (stable identity per dtype
    pair, so cuda.compute builds one kernel per pair). The output type is set by
    the return annotation."""

    def _nz(x):
        return 1 if x != 0 else 0

    _nz.__annotations__ = {"x": in_type, "return": out_type}
    return _nz


def _nonzero_for_reduce(input_data, out_dtype):
    """Fuse the ``!= 0`` map into a downstream segmented_reduce: a
    TransformIterator that maps each element to {0, 1} on the fly, so the
    reduction never sees (or allocates) a materialised ``mapped`` buffer.
    Equivalent to ``(input_data != 0).astype(out_dtype)`` without the buffer.
    (Real/integer inputs only — complex inputs keep the host-side CuPy map,
    since a complex-valued cuda.compute iterator risks NVRTC failures.)
    """
    return TransformIterator(
        input_data,
        _make_is_nonzero(input_data.dtype.type, np.dtype(out_dtype).type),
    )


def segmented_sort(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    ascending,
    stable,
):
    from cuda.compute import SortOrder, segmented_sort

    cupy_nplike = Cupy.instance()
    cp = cupy_nplike._module

    # Ensure offsets are int64 as expected by segmented_sort
    if offsets.dtype != cp.int64:
        offsets = offsets.astype(cp.int64, copy=False)

    num_segments = offsetslength - 1
    num_items = int(offsets[-1]) if len(offsets) > 0 else 0

    start_o, end_o = make_segment_views(offsets)

    order = SortOrder.ASCENDING if ascending else SortOrder.DESCENDING

    segmented_sort(
        d_in_keys=fromptr,
        d_out_keys=toptr,
        d_in_values=None,
        d_out_values=None,
        num_items=num_items,
        num_segments=num_segments,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        order=order,
        stream=None,
    )


def awkward_reduce_argmax(
    result,
    input_data,
    offsets_data,
    starts,
    outlength,
):
    index_dtype = normalize_index_dtype(offsets_data.dtype)

    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_argmax(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        if start_idx == end_idx:
            return -1
        # return a global index
        return np.argmax(segment) + start_idx

    segment_ids = CountingIterator(index_dtype(0))

    # TODO: replace with segmented_reduce once available/fixed in CCCL
    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_argmax,
        num_items=outlength,
    )


@cuda.jit(device=True)
def lexargmax_complex(values, start_idx, end_idx):
    if start_idx == end_idx:
        return np.int64(-1)

    best_idx = start_idx
    best = values[start_idx]

    i = start_idx + 1
    while i < end_idx:
        value = values[i]
        if value.real > best.real or (
            value.real == best.real and value.imag > best.imag
        ):
            best = value
            best_idx = i
        i += 1

    return best_idx


def awkward_reduce_argmax_complex(
    result,
    input_data,
    offsets_data,
    outlength,
):
    complex_dtype = infer_complex_dtype(input_data.dtype)
    input_complex = input_data.view(complex_dtype)

    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_argmax(segment_id):
        return lexargmax_complex(
            input_complex,
            start_o[segment_id],
            end_o[segment_id],
        )

    segment_ids = CountingIterator(index_dtype(0))

    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_argmax,
        num_items=outlength,
    )


def awkward_reduce_argmin(
    result,
    input_data,
    offsets_data,
    starts,
    outlength,
):
    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_argmin(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        if start_idx == end_idx:
            return -1
        # return a global index
        return np.argmin(segment) + start_idx

    segment_ids = CountingIterator(index_dtype(0))

    # TODO: replace with segmented_reduce once available/fixed in CCCL
    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_argmin,
        num_items=outlength,
    )


@cuda.jit(device=True)
def lexargmin_complex(values, start_idx, end_idx):
    if start_idx == end_idx:
        return np.int64(-1)

    best_idx = start_idx
    best = values[start_idx]

    i = start_idx + 1
    while i < end_idx:
        value = values[i]

        if value.real < best.real or (
            value.real == best.real and value.imag < best.imag
        ):
            best = value
            best_idx = i
        i += 1

    return best_idx


def awkward_reduce_argmin_complex(
    result,
    input_data,
    offsets_data,
    outlength,
):
    index_dtype = normalize_index_dtype(offsets_data.dtype)

    complex_dtype = infer_complex_dtype(input_data.dtype)
    input_complex = input_data.view(complex_dtype)

    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_argmin(segment_id):
        return lexargmin_complex(
            input_complex,
            start_o[segment_id],
            end_o[segment_id],
        )

    segment_ids = CountingIterator(index_dtype(0))

    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_argmin,
        num_items=outlength,
    )


def awkward_reduce_sum(
    result,
    input_data,
    offsets_data,
    outlength,
):
    d_input = _widen_for_reduce(input_data, result.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    h_init = np.asarray(0, dtype=result.dtype)

    segmented_reduce(
        d_in=d_input,
        d_out=result,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.PLUS,
        h_init=h_init,
    )


def awkward_reduce_sum_bool(
    result,
    input_data,
    offsets_data,
    outlength,
):
    # ak.any semantics: result is True for a bin iff any element is non-zero.
    # If we feed wider-than-bool input straight into a MAX reduction and then
    # truncate to int8 on store, an input like [256, 512, ...] reduces to MAX
    # = 512 which truncates to 0 → spurious False. Map to {0, 1} first, then
    # MAX = OR. Symmetric to awkward_reduce_prod_bool (MIN = AND).
    # Fuse the {0, 1} map into the reduction (no materialised `mapped` buffer).
    # A bool array is already {0, 1}, so a free int8 view suffices.
    if input_data.dtype == cp.bool_:
        d_in = input_data.view(cp.int8)
    else:
        d_in = _nonzero_for_reduce(input_data, cp.int8)

    d_out = result.view(cp.int8) if result.dtype == cp.bool_ else result
    start_o, end_o = make_segment_views(offsets_data)
    h_init = np.asarray(0, dtype=cp.int8)  # identity for MAX over {0, 1}

    segmented_reduce(
        d_in=d_in,
        d_out=d_out,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.MAXIMUM,
        h_init=h_init,
    )


def awkward_reduce_sum_bool_complex(
    result,
    input_data,
    offsets_data,
    outlength,
):
    complex_dtype = infer_complex_dtype(input_data.dtype)
    input_complex = input_data.view(complex_dtype)

    d_out = result.view(cp.int8) if result.dtype == cp.bool_ else result

    # Vectorised CuPy: a fresh `is_nonzero_complex` closure each call leaks
    # `mapped_data` via unary_transform's op-identity-keyed cache. A complex
    # value is non-zero iff its real or imaginary part is non-zero.
    mapped_data = (input_complex != 0).astype(cp.int8)

    start_o, end_o = make_segment_views(offsets_data)
    h_init = np.asarray(0, dtype=cp.int8)  # Identity for OR is False

    segmented_reduce(
        d_in=mapped_data,
        d_out=d_out,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.MAXIMUM,
        h_init=h_init,
    )


def awkward_reduce_sum_int32_bool_64(
    result,
    input_data,
    offsets_data,
    outlength,
):
    d_input = _widen_for_reduce(input_data, result.dtype)
    start_o, end_o = make_segment_views(offsets_data)
    h_init = np.asarray(0, dtype=result.dtype)

    segmented_reduce(
        d_in=d_input,
        d_out=result,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.PLUS,
        h_init=h_init,
    )


def awkward_reduce_sum_int64_bool_64(
    result,
    input_data,
    offsets_data,
    outlength,
):
    d_input = _widen_for_reduce(input_data, result.dtype)
    start_o, end_o = make_segment_views(offsets_data)
    h_init = np.asarray(0, dtype=result.dtype)

    segmented_reduce(
        d_in=d_input,
        d_out=result,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.PLUS,
        h_init=h_init,
    )


def awkward_reduce_sum_complex(
    result,
    input_data,
    offsets_data,
    outlength,
):
    complex_dtype = infer_complex_dtype(input_data.dtype)
    input_complex = input_data.view(complex_dtype)
    result_complex = result.view(complex_dtype)

    start_o, end_o = make_segment_views(offsets_data)

    h_init = np.asarray(0, dtype=complex_dtype)

    def sum_op(a, b):
        return a + b

    segmented_reduce(
        d_in=input_complex,
        d_out=result_complex,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=sum_op,
        h_init=h_init,
    )


# `awkward_reduce_sum_bool_complex` (above) handles both float32- and
# float64-interleaved inputs via `infer_complex_dtype`. The previous
# `awkward_reduce_sum_bool_complex64_64` / `_complex128_64` specialisations
# were verbatim duplicates with a hardcoded complex dtype — dispatch now
# routes both names to the generic implementation in _backends/cupy.py.


def awkward_reduce_prod(
    result,
    input_data,
    offsets_data,
    outlength,
):
    d_input = _widen_for_reduce(input_data, result.dtype)

    start_o, end_o = make_segment_views(offsets_data)
    h_init = np.asarray(1, dtype=result.dtype)

    def prod_op(a, b):
        return a * b

    segmented_reduce(
        d_in=d_input,
        d_out=result,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=prod_op,
        h_init=h_init,
    )


def awkward_reduce_prod_complex(
    result,
    input_data,
    offsets_data,
    outlength,
):
    complex_dtype = infer_complex_dtype(input_data.dtype)

    input_complex = input_data.view(complex_dtype)
    result_complex = result.view(complex_dtype)
    start_o, end_o = make_segment_views(offsets_data)

    h_init = np.asarray(1.0 + 0.0j, dtype=complex_dtype)

    def prod_op(a, b):
        return a * b

    segmented_reduce(
        d_in=input_complex,
        d_out=result_complex,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=prod_op,
        h_init=h_init,
    )


def awkward_reduce_prod_bool(
    result,
    input_data,
    offsets_data,
    outlength,
):
    # ak.all semantics: result is True for a bin iff every element is non-zero.
    # We must NOT use integer multiplication here — for an int64 input array,
    # the running product overflows mod 2^64 and frequently collapses to 0
    # even when every element is non-zero, giving a spurious False.
    # Instead, map each element to {0, 1} once and reduce with MIN (= AND).
    # This mirrors awkward_reduce_sum_bool's MAX-over-{0,1} for ak.any.
    # Fuse the {0, 1} map into the reduction (no materialised `mapped` buffer).
    if input_data.dtype == cp.bool_:
        d_in = input_data.view(cp.int8)
    else:
        d_in = _nonzero_for_reduce(input_data, cp.int8)

    d_out = result.view(cp.int8) if result.dtype == cp.bool_ else result
    start_o, end_o = make_segment_views(offsets_data)
    h_init = np.asarray(1, dtype=cp.int8)  # identity for MIN over {0, 1}

    segmented_reduce(
        d_in=d_in,
        d_out=d_out,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.MINIMUM,
        h_init=h_init,
    )


def awkward_reduce_prod_bool_complex(
    result,
    input_data,
    offsets_data,
    outlength,
):
    complex_dtype = infer_complex_dtype(input_data.dtype)
    input_complex = input_data.view(complex_dtype)

    # Vectorised CuPy: a fresh `is_nonzero` closure each call leaks `mapped_input`
    # via unary_transform's op-identity-keyed cache.
    mapped_input = (input_complex != 0).astype(cp.int8)

    start_o, end_o = make_segment_views(offsets_data)
    d_result = result.view(cp.int8)

    h_init = np.asarray(1, dtype=cp.int8)

    def prod_op(a, b):
        return a * b

    segmented_reduce(
        d_in=mapped_input,
        d_out=d_result,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=prod_op,
        h_init=h_init,
    )


def awkward_reduce_max(
    result,
    input_data,
    offsets_data,
    outlength,
    identity,
):
    start_o, end_o = make_segment_views(offsets_data)

    h_init = np.asarray(identity, dtype=input_data.dtype)

    # def max_op(a, b):
    #     return a if a > b else b

    segmented_reduce(
        d_in=input_data,
        d_out=result,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.MAXIMUM,
        h_init=h_init,
    )


def awkward_reduce_max_complex(
    result,
    input_data,
    offsets_data,
    outlength,
    identity,
):
    complex_dtype = infer_complex_dtype(input_data.dtype)
    input_complex = input_data.view(complex_dtype)
    result_complex = result.view(complex_dtype)

    start_o, end_o = make_segment_views(offsets_data)

    h_init = np.asarray(identity + 0.0j, dtype=complex_dtype)

    def lex_max_op(a, b):
        if a.real > b.real:
            return a
        if a.real == b.real and a.imag > b.imag:
            return a
        return b

    segmented_reduce(
        d_in=input_complex,
        d_out=result_complex,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=lex_max_op,
        h_init=h_init,
    )


def awkward_reduce_min(
    result,
    input_data,
    offsets_data,
    outlength,
    identity,
):
    start_o, end_o = make_segment_views(offsets_data)

    h_init = np.asarray(identity, dtype=input_data.dtype)

    segmented_reduce(
        d_in=input_data,
        d_out=result,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.MINIMUM,
        h_init=h_init,
    )


def awkward_reduce_min_complex(
    result,
    input_data,
    offsets_data,
    outlength,
    identity,
):
    complex_dtype = infer_complex_dtype(input_data.dtype)
    input_complex = input_data.view(complex_dtype)
    result_complex = result.view(complex_dtype)

    start_o, end_o = make_segment_views(offsets_data)

    h_init = np.asarray(identity + 0.0j, dtype=complex_dtype)

    def lex_min_op(a, b):
        if a.real < b.real:
            return a
        if a.real == b.real and a.imag < b.imag:
            return a
        return b

    segmented_reduce(
        d_in=input_complex,
        d_out=result_complex,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=lex_min_op,
        h_init=h_init,
    )


def awkward_reduce_count_64(
    result,
    offsets_data,
    _outlength,
):
    result[:] = offsets_data[1:] - offsets_data[:-1]


def awkward_reduce_countnonzero(
    result,
    input_data,
    offsets_data,
    outlength,
):
    # Temporary workaround for bool instability
    if input_data.dtype == cp.bool_:
        input_data = input_data.view(cp.int8)

    # Fuse the {0, 1} map into the reduction (no materialised `mapped` buffer);
    # output dtype is the result dtype so PLUS counts the non-zero elements.
    d_in = _nonzero_for_reduce(input_data, result.dtype)

    start_o, end_o = make_segment_views(offsets_data)
    h_init = np.asarray(0, dtype=result.dtype)

    segmented_reduce(
        d_in=d_in,
        d_out=result,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.PLUS,
        h_init=h_init,
    )


def awkward_reduce_countnonzero_complex(
    result,
    input_data,
    offsets_data,
    outlength,
):
    # Complex values arrive as a flat float32/float64 array of length 2*N
    # (real/imag interleaved). Re-view into complex dtype for reduction.

    complex_dtype = infer_complex_dtype(input_data.dtype)
    input_complex = input_data.view(complex_dtype)

    # Vectorised CuPy map (mirrors awkward_reduce_sum_bool_complex): a fresh
    # `segment_reduce_countnonzero` closure each call leaked `result` via
    # unary_transform's op-identity-keyed cache. Count non-zero elements per
    # segment = segmented PLUS over the {0, 1} map (a complex value is non-zero
    # iff its real or imaginary part is non-zero). segmented_reduce caches on
    # the compiled signature, not op identity, so nothing accumulates per call.
    mapped = (input_complex != 0).astype(result.dtype)

    start_o, end_o = make_segment_views(offsets_data)
    h_init = np.asarray(0, dtype=result.dtype)

    segmented_reduce(
        d_in=mapped,
        d_out=result,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.PLUS,
        h_init=h_init,
    )


# Overlays a mask onto an index array: masked positions become -1, unmasked positions keep their original index value.
def awkward_IndexedArray_overlay_mask(toindex, mask, fromindex, length):
    # Vectorised CuPy: the `transform` closure captured the device arrays
    # `mask`/`fromindex`, defeating cuda.compute's build-result cache so every
    # call pinned a fresh `toindex` (see test_4056_cuda_kernel_memory).
    if length == 0:
        return
    dtype = toindex.dtype.type
    toindex[:length] = cp.where(
        mask[:length].astype(cp.bool_), dtype(-1), fromindex[:length]
    )


# Skips masked (-1) entries and packs remaining valid entries into nextcarry, tracking where
# each ended up in outindex. Builds nextoffsets[j+1] = cumulative count of valid entries in
# segments 0..j as defined by the offsets array.
#
# Example:
# index    = [3, -1, 5, -1, 2, -1, 4]
# offsets  = [0, 4, 7]             (2 segments: positions 0-3 and 4-6)
# outlength = 2
#
# nextcarry   = [3, 5, 2, 4]       (valid index values, compacted)
# nextoffsets = [0, 2, 4]          (segment 0 has 2 valid, segment 1 has 2 valid)
# outindex    = [0, -1, 1, -1, 2, -1, 3]  (position in nextcarry, or -1 if masked)
def awkward_IndexedArray_reduce_next_64(
    nextcarry, nextoffsets, outindex, index, offsets, outlength
):
    nextoffsets[0] = 0
    if outlength == 0:
        return

    index_length = int(offsets[outlength])
    if index_length == 0:
        nextoffsets[1 : outlength + 1] = 0
        return

    idx_dtype = index.dtype
    valid = (index[:index_length] >= 0).astype(idx_dtype)
    scan = cp.empty(index_length, dtype=idx_dtype)
    inclusive_scan(
        d_in=valid,
        d_out=scan,
        op=lambda a, b: a + b,
        init_value=cp.array([0], dtype=idx_dtype),
        num_items=index_length,
    )

    def scatter_and_fill(i):
        if index[i] >= 0:
            k = scan[i] - 1
            nextcarry[k] = index[i]
            return k
        return -1

    unary_transform(
        d_in=CountingIterator(idx_dtype.type(0)),
        d_out=outindex,
        op=scatter_and_fill,
        num_items=index_length,
    )

    off_dtype = offsets.dtype.type

    def fill_nextoffsets(j):
        stop = offsets[j + 1]
        nextoffsets[j + 1] = idx_dtype.type(0) if stop == 0 else scan[stop - 1]
        return off_dtype(0)

    unary_transform(
        d_in=CountingIterator(off_dtype(0)),
        d_out=DiscardIterator(),
        op=fill_nextoffsets,
        num_items=outlength,
    )


# For each valid (non-negative) entry at position i, records the number of null (negative) entries
# that appeared before it. The k-th valid entry gets nextshifts[k] = count of nulls before position i.
# For example, für index = [0, 1, 2, -1, 3, -1, 4] → nextshifts = [0, 0, 0, 1, 2].
def awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64(nextshifts, index, length):
    if length == 0:
        return

    index_slice = index[:length]

    # cumsum of (index < 0) gives the running null count at each position.
    # this is basically equivalent to calling cuda.compute.inclusive_scan on index_slice < 0
    null_cumsum = cp.cumsum(index_slice < 0)
    _ = cp.empty(length, dtype=cp.int64)

    def scatter(i):
        null_count = null_cumsum[i]
        if index_slice[i] >= 0:
            nextshifts[i - null_count] = null_count  # output slot = i - null_count
        # return a dummy value otherwise
        return cp.int64(0)

    indices = CountingIterator(cp.int64(0))
    unary_transform(d_in=indices, d_out=_, op=scatter, num_items=length)


# Packs valid entries (where (mask[i] != 0) == validwhen) into tocarry in order.
# Examples:
# mask = [0, 1, 0, 1, 1], validwhen=True  → tocarry = [1, 3, 4]
# mask = [0, 1, 0, 1, 1], validwhen=False → tocarry = [0, 2]
# mask = [0, 1, 0, 1, 1, -1, 1], validwhen=True → tocarry = [1, 3, 4, 5, 6]
def awkward_ByteMaskedArray_getitem_nextcarry(tocarry, mask, length, validwhen):
    if length == 0:
        return

    # valid = ((mask[:length] != 0) == validwhen)
    # valid[i] is 1 when the masked element passes the validwhen condition.

    # get the indices of the valid entries using cp.nonzero
    valid_indices = cp.nonzero((mask[:length] != 0) == validwhen)[0]
    # in case tocarry is not exactly the right size, allocate it in two steps like this
    tocarry[: len(valid_indices)] = valid_indices


# Counts null (invalid) entries: positions where (mask[i] != 0) != validwhen.
# Examples:
# mask = [0, 1, 0, 1, 1], validwhen=True  → numnull = 2  (positions 0 and 2 are null)
# mask = [0, 1, 0, 1, 1], validwhen=False → numnull = 3  (positions 1, 3 and 4 are null)
def awkward_ByteMaskedArray_numnull(numnull, mask, length, validwhen):
    numnull[0] = cp.count_nonzero((mask[:length] != 0) != validwhen)


# Broadcasts a single jagged offset array across all rows of a regular array
# Example:
# singleoffsets = [0, 2, 5], regularsize = 2, regularlength = 3
# multistarts = [0, 2, 0, 2, 0, 2]
# multistops  = [2, 5, 2, 5, 2, 5]
def awkward_RegularArray_getitem_jagged_expand(
    multistarts, multistops, singleoffsets, regularsize, regularlength
):
    if regularlength == 0 or regularsize == 0:
        return

    # Reshape as (regularlength, regularsize) views (no copy) and broadcast-assign
    # singleoffsets[:-1] / singleoffsets[1:] across all rows.
    multistarts.reshape(regularlength, regularsize)[:] = singleoffsets[:regularsize]
    multistops.reshape(regularlength, regularsize)[:] = singleoffsets[
        1 : regularsize + 1
    ]


# THIS KERNEL IS NOT USED (just for archive)
# Fills a tagged index for one union type: assigns a constant tag and
# sequential index into each segment defined by the starts/counts ranges
# Example input:
# tmpstarts = [0, 3], tag = 1, fromcounts = [3, 2]
# Example output:
# totags  = [1, 1, 1, 1, 1]
# toindex = [0, 1, 2, 0, 1]
# also, the tmpstarts get rewritten with stops: tmpstarts = [3, 5]
def awkward_UnionArray_nestedfill_tags_index(
    totags, toindex, tmpstarts, tag, fromcounts, length
):
    if length == 0:
        return

    starts = tmpstarts[:length]
    counts = fromcounts[:length]

    # Total span of the output arrays we need to touch:
    # the last segment's start + its count gives the furthest written position
    total_size = int(starts[length - 1]) + int(counts[length - 1])

    if total_size == 0:
        return

    # +1 at each segment start, -1 just past each segment end.
    # cumsum of this will later yield 1 inside any covered range, 0 in gaps.
    diff = cp.zeros(total_size + 1, dtype=cp.int8)

    def scatter_and_update(i):
        start = starts[i]
        count = counts[i]
        # Mark this segment's range in the difference array
        diff[start] += cp.int8(1)
        diff[start + count] -= cp.int8(1)
        # update tmpstarts (for the next call of this kernel (for a different union type))?
        tmpstarts[i] = start + count
        return 0

    # Scatter segment's ranges and update tmpstarts
    unary_transform(
        d_in=CountingIterator(cp.int64(0)),
        d_out=DiscardIterator(),
        op=scatter_and_update,
        num_items=length,
    )

    # coverage[j] == 1 if position j falls inside any segment's range, 0 otherwise
    coverage = cp.cumsum(diff[:total_size])

    # scan[j] == local index of element j within its segment
    # Since it's a cumsum, the first index starts from 1, 2, 3 ...
    # so we'll have to -1 before writing it in toindex
    scan = cp.cumsum(coverage, dtype=cp.int64)

    def fill(j):
        if coverage[j]:
            # Mark this position as belonging to the current tag
            totags[j] = tag
            toindex[j] = scan[j] - 1
        return 0

    unary_transform(
        d_in=CountingIterator(cp.int64(0)),
        d_out=DiscardIterator(),
        op=fill,
        num_items=total_size,
    )


# For each position i where fromtags[i] == fromwhich, sets totags[i] = towhich and
# toindex[i] = fromindex[i] + base. Other positions are left unchanged.
# Example:
# fromtags  = [0, 1, 0, 1, 0], fromindex = [0, 0, 1, 1, 2]
# fromwhich=1, towhich=2, base=10
# totags  = [0, 2, 0, 2, 0]
# toindex = [0, 10, 1, 11, 2]
def awkward_UnionArray_simplify_one(
    totags, toindex, fromtags, fromindex, towhich, fromwhich, length, base
):
    if length == 0:
        return

    # Vectorised CuPy: the `transform` closure captured the device arrays
    # `fromtags`/`fromindex`, defeating cuda.compute's cache and leaking per
    # call. Positions where the tag does not match are left untouched.
    sel = cp.nonzero(fromtags[:length] == fromwhich)[0]
    totags[sel] = towhich
    toindex[sel] = fromindex[sel] + base


# producing a carry index that maps each output element back to its position in the original content
# Example input:
# fromoffsets = [0, 3, 5], fromstarts = [10, 20], fromstops = [13, 22], lencontent = 25
# Example output:
# i=0: range [10, 13) → [10, 11, 12]
# i=1: range [20, 22) → [20, 21]
# tocarry = [10, 11, 12, 20, 21]
def awkward_ListArray_broadcast_tooffsets(
    tocarry, fromoffsets, offsetslength, fromstarts, fromstops, lencontent
):
    if offsetslength <= 1:
        return

    length = offsetslength - 1
    starts = fromstarts[:length]
    stops = fromstops[:length]
    # counts[i] = how many elements list i should have
    counts = fromoffsets[1:offsetslength] - fromoffsets[:length]

    _K = "awkward_ListArray_broadcast_tooffsets"
    if cp.any((starts != stops) & (stops > lencontent)):
        raise ValueError(f"stops[i] > len(content) in compiled CUDA code ({_K})")
    if cp.any(counts < 0):
        raise ValueError(
            f"broadcast's offsets must be monotonically increasing in compiled CUDA code ({_K})"
        )
    if cp.any(stops - starts != counts):
        raise ValueError(f"cannot broadcast nested list in compiled CUDA code ({_K})")

    # For each segment i, write the content indices starts[i], starts[i]+1, ..., stops[i]-1
    # into the contiguous output slice tocarry[fromoffsets[i] : fromoffsets[i+1]].
    # Vectorised CuPy: the `fill_list` closure captured the device arrays
    # `starts`/`stops`/`fromoffsets`, defeating cuda.compute's cache and leaking
    # `tocarry` per call. Output positions are contiguous in
    # [fromoffsets[0], fromoffsets[length]); for output slot p = fromoffsets[i] + m
    # the value is starts[i] + m.
    total = int(counts.sum())
    if total == 0:
        return
    starts_per = cp.repeat(starts, counts)
    base_per = cp.repeat(fromoffsets[:length], counts)
    begin = int(fromoffsets[0])
    within = cp.arange(total, dtype=tocarry.dtype) - (base_per - begin).astype(
        tocarry.dtype, copy=False
    )
    tocarry[begin : begin + total] = (
        starts_per.astype(tocarry.dtype, copy=False) + within
    )


# For each segment i, it fills toindex with the local position of each element within that segment — i.e. 0, 1, 2, ...
# Example:
# offsets = [0, 3, 5]
# toindex = [0, 1, 2, 0, 1]
def awkward_ListArray_localindex(toindex, offsets, length):
    if length == 0:
        return

    # Vectorised CuPy: the `fill` closure captured the device arrays
    # `starts`/`stops`/`toindex`, defeating cuda.compute's cache and leaking per
    # call. This is the GENERIC ListArray kernel: starts/stops are consecutive
    # slices of `offsets` but are NOT guaranteed monotonic — stops[i] < starts[i]
    # is an empty list (the original `for j in range(start, stop)` writes
    # nothing), and ranges may be non-contiguous, so write each segment's local
    # positions 0..count-1 to absolute positions [start, stop) via a scatter.
    # Work in signed int64 to avoid unsigned underflow turning empty lists into
    # ~2**32-long ranges, and clamp empty/negative-length lists to zero.
    starts = offsets[:length].astype(cp.int64, copy=False)
    stops = offsets[1 : length + 1].astype(cp.int64, copy=False)
    counts = cp.maximum(stops - starts, cp.int64(0))
    total = int(counts.sum())
    if total == 0:
        return
    grp = cp.zeros(length, dtype=cp.int64)
    grp[1:] = cp.cumsum(counts)[:-1]
    local = cp.arange(total, dtype=cp.int64) - cp.repeat(grp, counts)
    positions = cp.repeat(starts, counts) + local
    toindex[positions] = local.astype(toindex.dtype, copy=False)


# Converts a ListArray's (starts, stops) pairs into offsets.
# tooffsets[0] = 0, tooffsets[i+1] = tooffsets[i] + (fromstops[i] - fromstarts[i])
# Example:
# fromstarts = [10, 20], fromstops = [13, 22], length = 2
# tooffsets = [0, 3, 5]
def awkward_ListArray_compact_offsets(tooffsets, fromstarts, fromstops, length):
    tooffsets[0] = 0
    if length == 0:
        return

    starts = fromstarts[:length]
    stops = fromstops[:length]

    if cp.any(stops < starts):
        raise ValueError(
            "stops[i] < starts[i] in compiled CUDA code (awkward_ListArray_compact_offsets)"
        )

    sizes = stops - starts

    # the same as `tooffsets[1 : length + 1] = cp.cumsum(sizes)`
    inclusive_scan(
        d_in=sizes,
        d_out=tooffsets[1 : length + 1],
        op=lambda a, b: a + b,
        init_value=cp.array([0], dtype=tooffsets.dtype),
        num_items=length,
    )


# For each list i, counts the number of n-combinations of its elements
# (with or without replacement) and builds an offsets array into tooffsets.
# totallen[0] is set to the total number of combinations across all lists.
#
# Example (n=2, replacement=False):
# starts=[0, 0, 0], stops=[2, 3, 4]
# sizes = [2, 3, 4]
# C(2,2)=1, C(3,2)=3, C(4,2)=6
# Then the output will be: tooffsets = [0, 1, 4, 10]
# totallen  = 10
def awkward_ListArray_combinations_length(
    totallen, tooffsets, n, replacement, starts, stops, length
):
    tooffsets[0] = 0
    if length == 0:
        totallen[0] = 0
        return

    def combinations_len(i):
        size = stops[i] - starts[i]
        if replacement:
            size = size + (n - 1)
        thisn = n
        if thisn > size:
            return 0
        elif thisn == size:
            return 1
        else:
            # C(size, n) == C(size, size-n), so use the smaller one
            # of the two to minimise the number of loop iterations
            if thisn * 2 > size:
                thisn = size - thisn

            # Compute C(size, thisn) = size! / (thisn! * (size-thisn)!) incrementally:
            # result = size * (size-1) * ... * (size-thisn+1) / thisn!
            result = size
            for j in range(2, thisn + 1):
                result = result * (size - j + 1)
                result = result // j
            return result

    # Compute the number of combinations for each list
    counts = cp.empty(length, dtype=tooffsets.dtype)
    unary_transform(
        d_in=CountingIterator(cp.int64(0)),
        d_out=counts,
        op=combinations_len,
        num_items=length,
    )

    # Convert counts to offsets:
    # tooffsets[i+1] = sum(counts[0..i])
    inclusive_scan(
        d_in=counts,
        d_out=tooffsets[1 : length + 1],
        op=lambda a, b: a + b,
        init_value=cp.array([0], dtype=tooffsets.dtype),
        num_items=length,
    )

    # Total number of combinations across all lists
    totallen[0] = tooffsets[length]


# For each list i, enumerates all n-combinations (with or without replacement)
# of its elements and writes the indices into n output carry arrays.
#
# tocarry_ptrs is a CuPy int64 array of length n holding raw device pointers;
# each pointer refers to a pre-allocated int64 array of length totallen.
#
# Example (n=2, replacement=False):
# starts=[0], stops=[3]  → elements [0,1,2]
# C(3,2) = 3 combinations in total
# combinations: (0,1),(0,2),(1,2)
#
# Output:
# tocarry_ptrs[0] → [0, 0, 1],  tocarry_ptrs[1] → [1, 2, 2]
# toindex: [3, 3]
def awkward_ListArray_combinations(
    tocarry_ptrs, toindex, fromindex, n, replacement, starts, stops, length
):
    if length == 0:
        return

    # Step 1: compute per-list combination counts (same as combinations_length!!)
    # TODO: we can just pass combination offsets directly in the future (from src/awkward/contents/listoffsetarray.py:1405)
    def combinations_len(i):
        size = stops[i] - starts[i]
        if replacement:
            size = size + (n - 1)
        thisn = n
        if thisn > size:
            return 0
        elif thisn == size:
            return 1
        else:
            if thisn * 2 > size:
                thisn = size - thisn
            result = size
            for j in range(2, thisn + 1):
                result = result * (size - j + 1)
                result = result // j
            return result

    counts = cp.empty(length, dtype=cp.int64)
    unary_transform(
        d_in=CountingIterator(cp.int64(0)),
        d_out=counts,
        op=combinations_len,
        num_items=length,
    )

    offsets = cp.empty(length + 1, dtype=cp.int64)
    offsets[0] = 0
    inclusive_scan(
        d_in=counts,
        d_out=offsets[1:],
        op=lambda a, b: a + b,
        init_value=cp.array([0], dtype=cp.int64),
        num_items=length,
    )

    totallen = int(offsets[length])
    if totallen == 0:
        return

    # Step 2: wrap raw pointers from tocarry_ptrs into CuPy arrays
    # raw int64 pointer values from tocarry_ptrs[k] can't be dereferenced inside a Numba closure, so
    # we need this intermediate step
    #
    # (the pointers themselves are allocated at src/awkward/contents/listoffsetarray.py:1456-1464)
    carry_arrays = []
    for k in range(n):
        ptr_val = int(tocarry_ptrs[k])
        mem = cp.cuda.UnownedMemory(ptr_val, totallen * 8, None)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        carry_arrays.append(cp.ndarray(totallen, dtype=cp.int64, memptr=memptr))  # pylint: disable=unexpected-keyword-arg

    # -------------------------------------------------------------------------
    # Step 3: fill carry_arrays[k] for each combination position k in turn.
    #
    # For each output slot g in [0, totallen):
    #
    #   a) Binary search offsets to find which source list i owns slot g,
    #      and compute the rank of this combination within that list
    #      (rank = g - offsets[i], i.e. the 0-based index among all combinations
    #      of list i in lexicographic order).
    #
    #   b) Unrank: decode the rank back into the actual combination tuple using
    #      a combinatorial number system. Iterating over positions pos=0..n-1,
    #      at each position scan forward through candidate values j, counting
    #      how many combinations start with values < j at this position
    #      (= C(effective_size-j-1, n-pos-1)). Subtract from remaining rank
    #      until we find the j where remaining < count — that j is the value
    #      at position pos.
    #
    #   c) Early exit: once pos==k we have the value for position k and write
    #      it to carry_k[g], skipping the rest of the unranking. This is why
    #      we do n separate passes (one per k) rather than one pass writing all
    #      n positions: each pass only needs to unrank up to position k.
    #
    #   d) Content index: add start (the list's base offset into content) to
    #      convert from a within-list index to an absolute content index.
    #      For replacement, subtract pos to undo the stars-and-bars shift.
    # -------------------------------------------------------------------------
    def make_pass(k, carry_k):
        def fill_pos(g):
            # a) Find source list i via binary search on offsets
            lo = 0
            hi = length - 1
            while lo < hi:
                mid = (lo + hi) >> 1
                if offsets[mid + 1] <= g:
                    lo = mid + 1
                else:
                    hi = mid
            list_i = lo
            start = starts[list_i]
            size = stops[list_i] - starts[list_i]
            rank = g - offsets[list_i]
            # For replacement use stars-and-bars effective size
            effective_size = size + n - 1 if replacement else size

            # b) Unrank: decode rank into the combination tuple
            lower = 0  # lower bound for j at each position (enforces ordering)
            remaining = rank
            for pos in range(n):
                for j in range(lower, effective_size - (n - pos - 1)):
                    # Count combinations where position pos has value j:
                    # = C(effective_size - j - 1, n - pos - 1)
                    top = effective_size - j - 1
                    choose = n - pos - 1
                    if choose == 0:
                        count = 1
                    else:
                        if choose * 2 > top:  # use smaller equivalent
                            choose = top - choose
                        c = top
                        for q in range(2, choose + 1):
                            c = c * (top - q + 1)
                            c = c // q
                        count = c
                    if remaining < count:
                        # c) j is the value at position pos
                        if pos == k:
                            # d) write absolute content index and exit early
                            carry_k[g] = (j - pos if replacement else j) + start
                            return 0
                        lower = j + 1  # next position must be >= j+1 (no repeat)
                        break
                    remaining -= count
            return 0

        return fill_pos

    # One parallel pass per combination position k
    for k in range(n):
        unary_transform(
            d_in=CountingIterator(cp.int64(0)),
            d_out=DiscardIterator(),
            op=make_pass(k, carry_arrays[k]),
            num_items=totallen,
        )

    toindex[:n] = totallen


def awkward_index_rpad_and_clip_axis0(toindex, target, length):
    """
    Fill ``toindex[0..target)`` with the identity mapping ``[0..shorter)``
    followed by ``target - shorter`` entries of ``-1``, where
    ``shorter = min(target, length)``.

    Called from ``Content._pad_none_axis0`` in
    ``src/awkward/contents/content.py``.
    """
    # Vectorised CuPy: a fresh `fill` closure each call leaks `toindex` via
    # unary_transform's op-identity-keyed cache. See test_4056_cuda_kernel_memory.
    dtype = toindex.dtype.type
    shorter = min(target, length)
    idx = cp.arange(target, dtype=dtype)
    toindex[:target] = cp.where(idx < shorter, idx, dtype(-1))


def awkward_index_rpad_and_clip_axis1(tostarts, tostops, target, length):
    """
    Fills `tostarts` and `tostops` with rpad/clip offsets for axis=1 lists.
    Each list is padded or clipped to length `target`.
    """

    # Vectorised CuPy: the `fill` closure captured the device array `tostarts`,
    # defeating cuda.compute's cache and leaking per call.
    if length == 0:
        return
    dtype = tostarts.dtype.type
    i = cp.arange(length, dtype=dtype)
    tostarts[:length] = i * dtype(target)
    tostops[:length] = i * dtype(target) + dtype(target)


def awkward_missing_repeat(
    outindex,
    index,
    indexlength,
    repetitions,
    regularsize,
):
    """
    Repeats an index array `repetitions` times, adjusting valid (non-negative)
    indices by an offset of `regularsize` per repetition. Missing values (-1)
    are preserved.

    Implemented with vectorised CuPy rather than a `unary_transform`: the
    transform's per-call `fill` closure captures `index`, which defeats
    cuda.compute's build-result cache so that every call pinned a fresh
    `outindex` through the cached iterators -- linear GPU-memory growth across
    repeated missing-slices (regression covered by
    tests-cuda/test_4056_cuda_kernel_memory.py::test_missing_repeat_memory).
    The plain-CuPy form allocates only short-lived temporaries that the memory
    pool reclaims, and is bit-for-bit equivalent to awkward_missing_repeat.cu:
    outindex[i * indexlength + j] = index[j] + (index[j] >= 0 ? i * regularsize : 0).
    """
    output_size = repetitions * indexlength
    if output_size == 0:
        return

    index_dtype = outindex.dtype.type

    # base[k] = index[k % indexlength] via a tile (no materialised gather index);
    # offset[k] = (k // indexlength) * regularsize, zeroed where base is missing
    # (< 0). Minimal-allocation form to stay within the per-call budget.
    base = cp.tile(index[:indexlength], repetitions).astype(index_dtype, copy=False)
    offset = cp.arange(output_size, dtype=index_dtype)
    offset //= indexlength
    offset *= index_dtype(regularsize)
    offset *= base >= 0
    base += offset
    outindex[:output_size] = base


# For each element in a regular array of `length` sublists of fixed `size`,
# writes its position within its sublist (0, 1, ..., size-1) into toindex.
# Example: size=3, length=2 → toindex = [0, 1, 2, 0, 1, 2]
def awkward_RegularArray_localindex(toindex, size, length):
    if length == 0 or size == 0:
        return

    # Vectorised CuPy: a fresh `fill` closure each call leaks `toindex` via
    # unary_transform's op-identity-keyed cache. See test_4056_cuda_kernel_memory.
    dtype = toindex.dtype.type
    toindex[: length * size] = cp.arange(length * size, dtype=dtype) % dtype(size)


# Broadcasts each element of fromadvanced across nextsize consecutive slots in toadvanced.
# Example: fromadvanced=[3, 7], nextsize=3 → toadvanced=[3, 3, 3, 7, 7, 7]
def awkward_RegularArray_getitem_next_range_spreadadvanced(
    toadvanced, fromadvanced, length, nextsize
):
    if length == 0 or nextsize == 0:
        return

    # Vectorised CuPy: the `fill` closure captured the device array
    # `fromadvanced`, defeating cuda.compute's cache and leaking per call.
    toadvanced[: length * nextsize] = cp.repeat(fromadvanced[:length], nextsize)


# Builds the carry index for slicing a RegularArray with a slice (start:stop:step).
# A RegularArray holds `length` sublists of fixed `size` in a flat content array.
# Applying a slice selects `nextsize` elements from each sublist starting at
# `regular_start` with stride `step`. The carry index maps each output position
# back to its source in the flat content:
#   tocarry[i*nextsize + j] = i*size + regular_start + j*step
# Example: size=6, length=3, slice 1:6:2 → regular_start=1, step=2, nextsize=3
#   tocarry = [1, 3, 5,  7, 9, 11,  13, 15, 17]
def awkward_RegularArray_getitem_next_range(
    tocarry, regular_start, step, length, size, nextsize
):
    if length == 0 or nextsize == 0:
        return

    # Vectorised CuPy (minimal allocation): a fresh `fill` closure each call
    # leaked `tocarry` via unary_transform's op-identity-keyed cache. Computes the
    # carry in place; the common nextsize==1 case needs a single buffer.
    dtype = tocarry.dtype.type
    n = length * nextsize
    out = cp.arange(n, dtype=dtype)
    if nextsize == 1:
        out *= dtype(size)
        out += dtype(regular_start)
    else:
        j = out % nextsize
        out //= nextsize
        out *= dtype(size)
        out += dtype(regular_start)
        j *= dtype(step)
        out += j
    tocarry[:n] = out


# Used when a RegularArray is indexed by two simultaneous advanced indices.
# fromadvanced[i] says which element of fromarray to use for row i, giving the
# intra-row column: col = fromarray[fromadvanced[i]].
# tocarry[i] = i*size + col  (flat content index)
# toadvanced[i] = i          (advanced-axis position for broadcast alignment)
# Example: size=4, fromarray=[1,3,0,2], fromadvanced=[0,2,1]
#   → tocarry=[1, 4, 11],  toadvanced=[0, 1, 2]
def awkward_RegularArray_getitem_next_array_advanced(
    tocarry, toadvanced, fromadvanced, fromarray, length, size
):
    # Vectorised CuPy rather than a `unary_transform`: the `fill_carry` closure
    # captures the device arrays `fromarray` and `fromadvanced`, which defeats
    # cuda.compute's build-result cache so every call pinned a fresh `tocarry`
    # (one buffer of `length`) through the cached iterators -- linear GPU-memory
    # growth across repeated advanced/missing slices (see
    # tests-cuda/test_4056_cuda_kernel_memory.py::test_missing_repeat_memory).
    # The plain-CuPy form allocates only short-lived temporaries that the pool
    # reclaims:
    #   tocarry[i]    = i * size + fromarray[fromadvanced[i]]
    #   toadvanced[i] = i
    if length == 0:
        return

    # Inputs/outputs may be longer than `length` (generated kernel tests pass
    # oversized buffers); operate only on the first `length` entries.
    dtype = tocarry.dtype.type
    i = cp.arange(length, dtype=dtype)
    tocarry[:length] = i * dtype(size) + fromarray[fromadvanced[:length]].astype(
        dtype, copy=False
    )
    toadvanced[:length] = cp.arange(length, dtype=toadvanced.dtype.type)


# Used when a RegularArray is indexed by a 1-D array (fromarray) of intra-row positions.
# Each of the `length` rows contributes `lenarray` output elements — one per entry in
# fromarray — so the output has shape (length, lenarray) laid out flat.
# tocarry[i*lenarray + j] = i*size + fromarray[j]  (flat content index for row i, column fromarray[j])
# toadvanced[i*lenarray + j] = j                   (which fromarray entry was used, for broadcast alignment)
# Example: length=2, size=5, fromarray=[1,3,0]
#   → tocarry=[1,3,0, 6,8,5],  toadvanced=[0,1,2, 0,1,2]
def awkward_RegularArray_getitem_next_array(
    tocarry, toadvanced, fromarray, length, lenarray, size
):
    if length == 0 or lenarray == 0:
        return

    # Vectorised CuPy (minimal allocation): the per-call `fill_both` closure
    # leaked via unary_transform's op-identity-keyed cache. Reuses one arange
    # buffer in place (i = k // lenarray) to stay within the per-call budget.
    dtype = tocarry.dtype.type
    n = length * lenarray
    k = cp.arange(n, dtype=dtype)
    j = k % lenarray
    toadvanced[:n] = j
    k //= lenarray
    k *= dtype(size)
    k += fromarray[j].astype(dtype, copy=False)
    tocarry[:n] = k


# Expands a carry index over a RegularArray of fixed-size sublists.
# For each carry element fromcarry[i] and intra-sublist position j,
# tocarry[i*size + j] = fromcarry[i]*size + j
# Example: fromcarry=[2, 0], size=3 → tocarry=[6, 7, 8, 0, 1, 2]
def awkward_RegularArray_getitem_carry(tocarry, fromcarry, lencarry, size):
    if lencarry == 0 or size == 0:
        return

    # Vectorised CuPy: the `fill` closure captured the device array `fromcarry`,
    # defeating cuda.compute's cache and leaking `tocarry` per call.
    dtype = tocarry.dtype.type
    k = cp.arange(lencarry * size, dtype=dtype)
    i = k // size
    j = k % size
    tocarry[: lencarry * size] = (
        fromcarry[i].astype(dtype, copy=False) * dtype(size) + j
    )


# Checks whether any two subranges of tmpptr (defined by fromstarts/fromstops) are equal.
# toequal[0] reflects the result of the last equal-length pair found
#
# Example:
#   tmpptr=[1,2,3,1,2,3,9], fromstarts=[0,3,6], fromstops=[3,6,7], length=3
#   Subranges 0 and 1 are both [1,2,3] → toequal[0] = True
def awkward_NumpyArray_subrange_equal(tmpptr, fromstarts, fromstops, length, toequal):
    n = length - 1
    # n<=1 → 0 pairs (triu_indices(n,k=1) is empty for n<2)
    if n <= 1:
        toequal[0] = cp.bool_(False)
        return
    lengths = fromstops[:n] - fromstarts[:n]

    # Enumerate all (i < ii) pairs
    i_idx, ii_idx = cp.triu_indices(n, k=1)
    pair_lengths_equal = lengths[i_idx] == lengths[ii_idx]

    has_valid = cp.any(pair_lengths_equal)  # 0-d GPU bool

    # Last equal-length pair: argmax on the reversed mask gives its distance from the end.
    # When all-False, argmax returns 0 and last_pos = num_pairs-1; has_valid corrects the result.
    num_pairs = len(i_idx)
    last_pos = (num_pairs - 1) - cp.argmax(pair_lengths_equal[::-1])

    si = fromstarts[i_idx[last_pos]]  # 0-d GPU int
    sii = fromstarts[ii_idx[last_pos]]
    li = lengths[i_idx[last_pos]]

    # Use len(tmpptr) as the arange upper bound
    N = len(tmpptr)
    j = cp.arange(N, dtype=fromstarts.dtype)
    in_range = j < li  # GPU broadcast
    safe_j = cp.where(
        in_range, j, fromstarts.dtype.type(0)
    )  # clamp out-of-range positions to 0
    differs = in_range & (tmpptr[si + safe_j] != tmpptr[sii + safe_j])

    toequal[:1] = (has_valid & ~cp.any(differs)).reshape(1)  # GPU→GPU, no sync


# Pads each variable-length sublist to a fixed target length, producing a flat output array.
#
# Layout:
#   fromptr      - flat source data
#   fromoffsets  - offsetslength boundary values; sublist k spans fromptr[fromoffsets[k]:fromoffsets[k+1]]
#   toptr        - flat output of size num_sublists * target
#
# For each output position q = k*target + j  (k = sublist index, j = intra-sublist position):
#   if j < count[k]:  toptr[q] = fromptr[fromoffsets[k] + j]   (copy)
#   else:             toptr[q] = 0                              (zero-fill)
#
# Example (target=3):
#   fromptr     = [10, 20, 30, 40, 50]
#   fromoffsets = [0, 2, 3, 5]          → sublists [10,20], [30], [40,50]
#   toptr       = [10, 20, 0, 30, 0, 0, 40, 50, 0]
def awkward_NumpyArray_pad_zero_to_length(
    fromptr, fromoffsets, offsetslength, target, toptr
):
    num_sublists = offsetslength - 1
    if num_sublists <= 0 or target == 0:
        return

    # Vectorised CuPy: the `fill` closure captured the device arrays
    # `fromptr`/`fromoffsets`, defeating cuda.compute's cache and leaking
    # `toptr` per call.
    dtype = toptr.dtype.type
    n_out = num_sublists * target
    if fromptr.size == 0:
        toptr[:n_out] = dtype(0)
        return
    q = cp.arange(n_out, dtype=cp.int64)
    k = q // target
    j = q % target
    start = fromoffsets[:num_sublists][k]
    count = fromoffsets[1 : num_sublists + 1][k] - start
    valid = j < count
    src = cp.where(valid, start + j, 0)
    toptr[:n_out] = cp.where(valid, fromptr[src].astype(dtype, copy=False), dtype(0))


# Same as awkward_NumpyArray_subrange_equal but for bool arrays.
# tmpptr is bool* (C++) / cupy.bool_ (Python), so values are always 0 or 1;
def awkward_NumpyArray_subrange_equal_bool(
    tmpptr, fromstarts, fromstops, length, toequal
):
    awkward_NumpyArray_subrange_equal(tmpptr, fromstarts, fromstops, length, toequal)


# Projects a jagged array through a mask: copies (starts_in[i], stops_in[i]) to the
# next output slot only for entries where index[i] >= 0 (i.e. not masked out).
# Example:
#   index      = [ 0, -1,  2, -1,  4]   (entries 1 and 3 are masked)
#   starts_in  = [10, 20, 30, 40, 50]
#   stops_in   = [12, 25, 33, 44, 51]
#   length     = 5
#
#   starts_out = [10, 30, 50]            (slots for entries 0, 2, 4)
#   stops_out  = [12, 33, 51]
def awkward_MaskedArray_getitem_next_jagged_project(
    index, starts_in, stops_in, starts_out, stops_out, length
):
    if length == 0:
        return
    mask = index[:length] >= 0
    indices = cp.where(mask)[0]
    n = len(indices)
    starts_out[:n] = starts_in[indices]
    stops_out[:n] = stops_in[indices]


# Builds a gather index to pad/clip a RegularArray from `size` to `target` elements per row.
# Output has length*target entries. For row i and slot j (q = i*target + j):
#   j < min(target, size)  →  i*size + j   (copy from source)
#   j >= min(target, size) →  -1            (pad with missing)
# When target <= size every slot maps to source (pure clip, no -1s).
# When target > size the last (target - size) slots per row are -1 (pure pad).
#
# Example size=3, target=5, length=2: toindex = [0, 1, 2, -1, -1,  3, 4, 5, -1, -1]
# Example size=5, target=3, length=2: toindex = [0, 1, 2,  5, 6, 7]
def awkward_RegularArray_rpad_and_clip_axis1(toindex, target, size, length):
    if length == 0 or target == 0:
        return
    # Vectorised CuPy: a fresh `fill` closure each call leaks `toindex` via
    # unary_transform's op-identity-keyed cache. See test_4056_cuda_kernel_memory.
    shorter = min(target, size)
    dtype = toindex.dtype.type
    q = cp.arange(length * target, dtype=dtype)
    i = q // target
    j = q % target
    toindex[: length * target] = cp.where(j < shorter, i * dtype(size) + j, dtype(-1))


# IS NOT USED (keep for archive). Takes too long to check for error cases.
# Normalizes indices for a regular array of size `size`:
# adds size to negative values, then checks bounds.
# Raises ValueError immediately if any index is still out of [0, size).
#
# Example (size=5): fromarray=[-1, 2, 0] → toarray=[4, 2, 0]
def awkward_RegularArray_getitem_next_array_regularize(
    toarray, fromarray, lenarray, size
):
    if lenarray == 0:
        return
    dtype = toarray.dtype.type

    def _regularize(v):
        return dtype(v + size) if v < 0 else dtype(v)

    unary_transform(
        d_in=fromarray[:lenarray],
        d_out=toarray[:lenarray],
        op=_regularize,
        num_items=lenarray,
    )
    if cp.any((toarray[:lenarray] < 0) | (toarray[:lenarray] >= size)):
        raise ValueError(
            "index out of range in compiled CUDA code (awkward_RegularArray_getitem_next_array_regularize)"
        )


# Builds a gather index for selecting one element per row from a RegularArray.
# A RegularArray stores `length` rows of exactly `size` elements in a flat buffer,
# so row i occupies flat positions [i*size, (i+1)*size).
# `at` is the column index to select (negative values wrap: at += size).
# tocarry[i] = i*size + at, which is then used to gather the chosen column from the flat data.
# Raises ValueError if at is out of [-size, size).
#
# Example size=5, at=2,  length=4: tocarry = [2, 7, 12, 17]
# Example size=5, at=-1, length=4: tocarry = [4, 9, 14, 19]  (last element of each row)
def awkward_RegularArray_getitem_next_at(tocarry, at, length, size):
    regular_at = int(at)
    if regular_at < 0:
        regular_at += size
    if regular_at < 0 or regular_at >= size:
        raise ValueError(
            "index out of range in compiled CUDA code (awkward_RegularArray_getitem_next_at)"
        )
    if length == 0:
        return
    # Vectorised CuPy: a fresh `_index` closure each call leaks `tocarry` via
    # unary_transform's op-identity-keyed cache. See test_4056_cuda_kernel_memory.
    dtype = tocarry.dtype.type
    tocarry[:length] = cp.arange(length, dtype=dtype) * dtype(size) + dtype(regular_at)


# IS NOT USED (keep for archive). Takes too long to check for error cases.
# Checks that all sublists have equal length and writes that length to size[0].
# Raises ValueError immediately if offsets decrease or sublists have different lengths.
#
# Example: fromoffsets=[0, 2, 4, 6] → size[0]=2
def awkward_ListOffsetArray_toRegularArray(size, fromoffsets, offsetslength):
    if offsetslength <= 1:
        size[:1] = cp.zeros(1, dtype=size.dtype)
        return
    n = offsetslength - 1
    counts = fromoffsets[1:offsetslength] - fromoffsets[:n]
    if cp.any(counts < 0):
        raise ValueError(
            "offsets must be monotonically increasing in compiled CUDA code (awkward_ListOffsetArray_toRegularArray)"
        )
    if not cp.all(counts == counts[0]):
        raise ValueError(
            "cannot convert to RegularArray because subarray lengths are not regular in compiled CUDA code (awkward_ListOffsetArray_toRegularArray)"
        )
    size[:1] = counts[:1].astype(size.dtype, copy=False)  # GPU→GPU


# Pads each sublist to at least target length (longer sublists kept as-is).
# Writes new sublist boundaries to tooffsets and total output length to tolength[0].
#
# Example (target=3): fromoffsets=[0,1,5,7] → tooffsets=[0,3,7,10], tolength[0]=10
def awkward_ListOffsetArray_rpad_length_axis1(
    tooffsets, fromoffsets, fromlength, target, tolength
):
    if fromlength == 0:
        tooffsets[:1] = cp.zeros(1, dtype=tooffsets.dtype)
        tolength[:1] = cp.zeros(1, dtype=tolength.dtype)
        return
    counts = fromoffsets[1 : fromlength + 1] - fromoffsets[:fromlength]
    padded = cp.maximum(counts, target)
    tooffsets[:1] = cp.zeros(1, dtype=tooffsets.dtype)
    inclusive_scan(
        d_in=padded,
        d_out=tooffsets[1 : fromlength + 1],
        op=OpKind.PLUS,
        init_value=None,
        num_items=fromlength,
    )
    tolength[:1] = tooffsets[fromlength : fromlength + 1].astype(
        tolength.dtype, copy=False
    )  # GPU→GPU, no sync


## NOT USED (revisit later)
# Builds a flat gather index for padding each sublist to at least `target` elements.
# The output size per row is max(len_i, target): longer sublists are kept in full,
# shorter ones are extended with -1 (missing).  toindex is sized to sum(max(len_i, target)).
# For output element q in row i at intra-row position j:
#   j < len_i  →  fromoffsets[i] + j   (copy actual element)
#   j >= len_i →  -1                    (pad with missing)
#
# Example (target=3): fromoffsets=[0,1,5,7], fromlength=3
#   row lengths: [1, 4, 2], padded: [3, 4, 3]
#   toindex = [0,-1,-1, 1,2,3,4, 5,6,-1]
def awkward_ListOffsetArray_rpad_axis1(toindex, fromoffsets, fromlength, target):
    if fromlength == 0 or len(toindex) == 0:
        return
    counts = fromoffsets[1 : fromlength + 1] - fromoffsets[:fromlength]
    starts = cp.empty(fromlength + 1, dtype=fromoffsets.dtype)
    padded = cp.maximum(counts, target).astype(starts.dtype, copy=False)
    starts[0] = 0
    inclusive_scan(
        d_in=padded,
        d_out=starts[1 : fromlength + 1],
        op=OpKind.PLUS,
        init_value=None,
        num_items=fromlength,
    )
    dtype = toindex.dtype.type

    # For each flat output index q, binary-searches starts to find which row i that position belongs to.
    # Computes the intra-row offset j = q - starts[i], and returns
    # either the source index (fromoffsets[i] + j) if within the original sublist length,
    # or -1 if it's a padding slot.
    def fill(q):
        lo = 0
        hi = fromlength - 1
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if starts[mid] <= q:
                lo = mid
            else:
                hi = mid - 1
        i = lo
        j = q - starts[i]
        if j < counts[i]:
            return dtype(fromoffsets[i] + j)
        return dtype(-1)

    unary_transform(
        d_in=CountingIterator(dtype(0)),
        d_out=toindex,
        op=fill,
        num_items=len(toindex),
    )


# Builds a flat gather index for padding/clipping each sublist to exactly `target` elements.
# Unlike rpad_axis1 the output size is always length*target (rows are never extended beyond target).
# For output element q in row i at intra-row slot j = q % target:
#   j < min(target, len_i)  →  fromoffsets[i] + j  (copy actual element)
#   j >= min(target, len_i) →  -1                   (pad with missing)
#
# Example (target=3): fromoffsets=[0,1,5,7], length=3
#   toindex = [0,-1,-1, 1,2,3, 5,6,-1]
def awkward_ListOffsetArray_rpad_and_clip_axis1(toindex, fromoffsets, length, target):
    if length == 0 or target == 0:
        return
    # Vectorised CuPy: a fresh `fill` closure each call leaks `toindex` via
    # unary_transform's op-identity-keyed cache. See test_4056_cuda_kernel_memory.
    dtype = toindex.dtype.type
    q = cp.arange(length * target, dtype=cp.int64)
    i = q // target
    j = q % target
    starts = fromoffsets[:length][i]
    rangeval = fromoffsets[1 : length + 1][i] - starts
    shorter = cp.minimum(rangeval, target)
    toindex[: length * target] = cp.where(
        j < shorter, (starts + j).astype(dtype, copy=False), dtype(-1)
    )


# Copies offsets[0:length+1] into offsetscopy and finds the maximum sublist length.
# maxcount[0] = max(offsets[i+1] - offsets[i] for i in range(length)), or 0 if length==0.
#
# Example: offsets=[0,3,3,7,8], length=4
#   offsetscopy = [0, 3, 3, 7, 8]
#   counts = [3, 0, 4, 1]  →  maxcount = [4]
def awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64(
    maxcount, offsetscopy, offsets, length
):
    offsetscopy[: length + 1] = offsets[: length + 1]
    if length > 0:
        counts = offsets[1 : length + 1] - offsets[:length]
        maxcount[:1] = cp.max(counts)
    else:
        maxcount[:1] = cp.zeros(1, dtype=maxcount.dtype)


# Writes the argsort of fromindex into tocarry.
# Used to reorder elements into the sorted order needed for nonlocal reductions.
#
# Example: fromindex=[2,0,3,1], length=4
#   tocarry = [1, 3, 0, 2]  (position of 0, 1, 2, 3 in fromindex)
def awkward_ListOffsetArray_local_preparenext_64(tocarry, fromindex, length):
    if length == 0:
        return
    tocarry[:length] = cp.argsort(fromindex[:length])


## NOT USED (revisit later)
# Gathers inneroffsets at positions given by outeroffsets:
# tooffsets[i] = inneroffsets[outeroffsets[i]] for i in [0, outeroffsetslen).
# Used when flattening a nested ListOffsetArray: the outer offsets index into
# the inner offset array to produce the combined flat offsets.
#
# Example: outeroffsets=[0,2,3], inneroffsets=[0,4,7,9,11]
#   tooffsets = [inneroffsets[0], inneroffsets[2], inneroffsets[3]] = [0, 9, 11]
def awkward_ListOffsetArray_flatten_offsets(
    tooffsets, outeroffsets, outeroffsetslen, inneroffsets
):
    if outeroffsetslen == 0:
        return
    tooffsets[:outeroffsetslen] = inneroffsets[outeroffsets[:outeroffsetslen]]


## NOT USED (revisit later)
# Rewrites fromoffsets to exclude None entries counted by noneindexes.
# tooffsets[i] = fromoffsets[i] - (number of nones in noneindexes[0:fromoffsets[i]])
# which equals the number of non-None elements before position fromoffsets[i].
#
# Computed via a single inclusive scan over the None mask, then indexed at each fromoffsets value.
#
# Example: noneindexes=[-1, 0, -1, 0, 0, -1, 0], fromoffsets=[0,2,3,5,7], length_offsets=5
#   none mask:        [1, 0, 1, 0, 0, 1, 0]
#   cumulative_nones: [0, 1, 1, 2, 2, 2, 3, 3]
#   tooffsets = fromoffsets - cumulative_nones[fromoffsets] = [0,2,3,5,7] - [0,1,2,2,3] = [0,1,1,3,4]
def awkward_ListOffsetArray_drop_none_indexes(
    tooffsets, noneindexes, fromoffsets, length_offsets, length_indexes
):
    if length_offsets == 0:
        return
    # cumulative_nones[k] = number of nones in noneindexes[0:k]
    cumulative_nones = cp.empty(length_indexes + 1, dtype=tooffsets.dtype)
    cumulative_nones[0] = 0
    if length_indexes > 0:
        none_mask = (noneindexes[:length_indexes] < 0).astype(
            cumulative_nones.dtype, copy=False
        )
        inclusive_scan(
            d_in=none_mask,
            d_out=cumulative_nones[1 : length_indexes + 1],
            op=OpKind.PLUS,
            init_value=None,
            num_items=length_indexes,
        )
    idx = fromoffsets[:length_offsets]
    tooffsets[:length_offsets] = (idx - cumulative_nones[idx]).astype(
        tooffsets.dtype, copy=False
    )


## NOT USED (revisit later)
# Validates that each non-empty row satisfies 0 <= starts[i] <= stops[i] <= lencontent.
# Empty rows (starts[i] == stops[i]) are skipped.
# Raises ValueError on the first failing row
#
# Example: starts=[0,2,5], stops=[2,5,8], lencontent=8  → no error
# Example: starts=[0,3], stops=[2,1], lencontent=8      → raises "start[i] > stop[i]"
def awkward_ListArray_validity(starts, stops, length, lencontent):
    if length == 0:
        return
    s = starts[:length]
    e = stops[:length]
    nonempty = s != e
    # Fuse all three condition checks into one cp.any() so the happy path
    # (no errors, the common case) only needs a single GPU→CPU sync instead of three.
    if not cp.any(nonempty & ((s > e) | (s < 0) | (e > lencontent))):
        return
    # Error path: identify which condition failed and find its first index.
    mask = nonempty & (s > e)
    if cp.any(mask):
        raise ValueError(f"start[i] > stop[i] at i={int(cp.argmax(mask))}")
    mask = nonempty & (s < 0)
    if cp.any(mask):
        raise ValueError(f"start[i] < 0 at i={int(cp.argmax(mask))}")
    mask = nonempty & (e > lencontent)
    raise ValueError(f"stop[i] > len(content) at i={int(cp.argmax(mask))}")


## NOT USED (revisit later)
# Pads each sublist to at least `target` elements, writing -1 for padding slots.
# Also writes tostarts/tostops describing the new (padded) layout.
# Unlike the ListOffsetArray variant, starts/stops are separate arrays rather than a single offsets array.
#
# For each row i with rangeval = stops[i] - starts[i]:
#   tostarts[i] = cumulative offset of row i in the output
#   tostops[i]  = tostarts[i] + max(rangeval, target)
#   toindex[tostarts[i]+j] = starts[i]+j  for j < rangeval  (copy from source)
#   toindex[tostarts[i]+j] = -1           for j >= rangeval  (padding)
#
# Example: starts=[0,3,3], stops=[3,3,5], target=4
#   tostarts=[0,4,8], tostops=[4,8,9]  (row 1 is empty → padded to 4; row 2 has 2 elems → stays 4 max(2,4)=4... wait)
#   Actually: max(3,4)=4, max(0,4)=4, max(2,4)=4 → tostarts=[0,4,8], tostops=[4,8,12]
#   toindex = [0,1,2,-1, -1,-1,-1,-1, 3,4,-1,-1]
def awkward_ListArray_rpad_axis1(
    toindex, fromstarts, fromstops, tostarts, tostops, target, length
):
    if length == 0:
        return
    counts = fromstops[:length] - fromstarts[:length]
    row_starts = cp.empty(length + 1, dtype=tostarts.dtype)
    row_starts[0] = 0
    inclusive_scan(
        d_in=cp.maximum(counts, target).astype(row_starts.dtype, copy=False),
        d_out=row_starts[1 : length + 1],
        op=OpKind.PLUS,
        init_value=None,
        num_items=length,
    )
    tostarts[:length] = row_starts[:length]
    tostops[:length] = row_starts[1 : length + 1]
    total = int(row_starts[length])
    if total == 0:
        return
    dtype = toindex.dtype.type

    def fill(q):
        lo = 0
        hi = length - 1
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if row_starts[mid] <= q:
                lo = mid
            else:
                hi = mid - 1
        i = lo
        j = q - row_starts[i]
        if j < counts[i]:
            return dtype(fromstarts[i] + j)
        return dtype(-1)

    unary_transform(
        d_in=CountingIterator(dtype(0)),
        d_out=toindex[:total],
        op=fill,
        num_items=total,
    )


# Computes the total output length for rpad_axis1 on a ListArray:
# tomin[0] = sum(max(target, stops[i] - starts[i])) for i in [0, lenstarts).
#
# Example: starts=[0,3,3], stops=[3,3,5], target=4
#   lengths=[3,0,2] → max with 4 → [4,4,4] → sum=12  →  tomin=[12]
def awkward_ListArray_rpad_and_clip_length_axis1(
    tomin, fromstarts, fromstops, target, lenstarts
):
    if lenstarts == 0:
        tomin[:1] = 0
        return
    counts = fromstops[:lenstarts] - fromstarts[:lenstarts]
    tomin[:1] = cp.sum(cp.maximum(counts, target)).astype(tomin.dtype, copy=False)


# Computes the minimum sublist length across all rows.
# tomin[0] = min(stops[i] - starts[i]) for i in [0, lenstarts).
#
# Example: starts=[0,3,5], stops=[3,5,6]
#   lengths=[3,2,1]  →  tomin=[1]
def awkward_ListArray_min_range(tomin, fromstarts, fromstops, lenstarts):
    if lenstarts == 0:
        return

    # Single-pass min(stops[i] - starts[i]) via ZipIterator + TransformIterator:
    def diff(pair):
        return pair.field_0 - pair.field_1

    dtype = fromstops.dtype
    zipped = ZipIterator(fromstops[:lenstarts], fromstarts[:lenstarts])
    lengths_iter = TransformIterator(zipped, diff)
    # should account for floats with `np.finfo` too?
    h_init = np.array([np.iinfo(dtype).max], dtype=dtype)
    if tomin.dtype == dtype:
        reduce_into(
            d_in=lengths_iter,
            d_out=tomin[:1],
            op=OpKind.MINIMUM,
            num_items=lenstarts,
            h_init=h_init,
        )
    else:
        result = cp.empty(1, dtype=dtype)
        reduce_into(
            d_in=lengths_iter,
            d_out=result,
            op=OpKind.MINIMUM,
            num_items=lenstarts,
            h_init=h_init,
        )
        tomin[:1] = result.astype(tomin.dtype, copy=False)


## NOT USED (revisit later)
# Broadcasts fromadvanced[i] across all positions in row i's output range [fromoffsets[i], fromoffsets[i+1]).
# toadvanced[fromoffsets[i] + j] = fromadvanced[i] for j in [0, fromoffsets[i+1] - fromoffsets[i]).
#
# Example: fromadvanced=[7,8,9], fromoffsets=[0,2,2,3]
#   toadvanced = [7,7, 9]  (row 1 is empty, row 2 has 1 element)
def awkward_ListArray_getitem_next_range_spreadadvanced(
    toadvanced, fromadvanced, fromoffsets, lenstarts
):
    if lenstarts == 0:
        return
    total = int(fromoffsets[lenstarts])
    if total == 0:
        return
    dtype = toadvanced.dtype.type

    def lookup(q):
        lo = 0
        hi = lenstarts - 1
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if fromoffsets[mid] <= q:
                lo = mid
            else:
                hi = mid - 1
        return dtype(fromadvanced[lo])

    unary_transform(
        d_in=CountingIterator(dtype(0)),
        d_out=toadvanced[:total],
        op=lookup,
        num_items=total,
    )


# Computes the total number of elements across all rows: fromoffsets[lenstarts] - fromoffsets[0].
#
# Example: fromoffsets=[0,3,3,5]  →  total=[5]
def awkward_ListArray_getitem_next_range_counts(total, fromoffsets, lenstarts):
    if lenstarts == 0:
        total[:1] = 0
        return
    total[:1] = fromoffsets[lenstarts].astype(total.dtype, copy=False) - fromoffsets[
        0
    ].astype(total.dtype, copy=False)


## NOT USED (revisit later)
# For each row i, selects the element at position `at` within that row.
# Negative `at` wraps relative to each row's own length: regular_at = at + (stops[i] - starts[i]).
# tocarry[i] = starts[i] + regular_at.
# Raises ValueError if any regular_at is out of [0, length_i).
#
# Example: starts=[0,3,5], stops=[3,5,6], at=-1
#   lengths=[3,2,1], regular_at=[2,1,0]  →  tocarry=[2,4,5]
def awkward_ListArray_getitem_next_at(tocarry, fromstarts, fromstops, lenstarts, at):
    if lenstarts == 0:
        return
    dtype = tocarry.dtype.type

    def compute(i):
        length = fromstops[i] - fromstarts[i]
        reg = length + at if at < 0 else at
        if reg < 0 or reg >= length:
            return dtype(-1)
        return dtype(fromstarts[i] + reg)

    unary_transform(
        d_in=CountingIterator(dtype(0)),
        d_out=tocarry[:lenstarts],
        op=compute,
        num_items=lenstarts,
    )
    if cp.any(tocarry[:lenstarts] < 0):
        i = int(cp.argmax(tocarry[:lenstarts] < 0))
        raise ValueError(f"index out of range at i={i}, at={at}")


# kSliceNone sentinel value (matches C++ kMaxInt64 + 1 = 2**63 - 1)
# also used in raw CUDA (at src/awkward/_connect/cuda/cuda_kernels/cuda_common.cu#L60)
_kSliceNone = 9223372036854775807


# an analogy of what is used in raw cuda kernels (awkward_regularize_rangeslice() inside cuda_common.cu)
def _make_count_row(fromstarts, fromstops, start, stop, step, dtype):
    def count_row(i):
        length = fromstops[i] - fromstarts[i]
        rs = start
        re = stop
        if step > 0:
            if rs == _kSliceNone:
                rs = 0
            elif rs < 0:
                rs = rs + length
            if rs < 0:
                rs = 0
            if rs > length:
                rs = length
            if re == _kSliceNone:
                re = length
            elif re < 0:
                re = re + length
            if re < 0:
                re = 0
            if re > length:
                re = length
            if re < rs:
                re = rs
            diff = re - rs
            if diff <= 0:
                return dtype(0)
            return dtype((diff + step - 1) // step)
        else:
            if rs == _kSliceNone:
                rs = length - 1
            elif rs < 0:
                rs = rs + length
            if rs < -1:
                rs = -1
            if rs > length - 1:
                rs = length - 1
            if re == _kSliceNone:
                re = -1
            elif re < 0:
                re = re + length
            if re < -1:
                re = -1
            if re > length - 1:
                re = length - 1
            if re > rs:
                re = rs
            diff = rs - re
            if diff <= 0:
                return dtype(0)
            neg_step = -step
            return dtype((diff + neg_step - 1) // neg_step)

    return count_row


# Computes total elements selected by slice(start, stop, step) across all rows.
# carrylength[0] = sum over i of the count of positions selected in row i of length
# fromstops[i] - fromstarts[i] by the range slice.
#
# Example: fromstarts=[0,3,5], fromstops=[3,5,6], start=kSliceNone, stop=kSliceNone, step=1
#   lengths=[3,2,1], all rows selected fully → carrylength=[6]
def awkward_ListArray_getitem_next_range_carrylength(
    carrylength, fromstarts, fromstops, lenstarts, start, stop, step
):
    if lenstarts == 0:
        carrylength[:1] = 0
        return

    # create an int64 type specifically, so that _make_count_row() doesn't break
    counts = cp.empty(lenstarts, dtype=cp.int64)
    unary_transform(
        d_in=CountingIterator(fromstarts.dtype.type(0)),
        d_out=counts,
        op=_make_count_row(fromstarts, fromstops, start, stop, step, cp.int64),
        num_items=lenstarts,
    )
    reduce_into(
        d_in=counts,
        d_out=carrylength[:1],
        op=OpKind.PLUS,
        num_items=lenstarts,
        h_init=np.array([0], dtype=counts.dtype),
    )


## NOT USED (revisit later) (1.76x slower than raw CUDA)
# Applies a range slice to a ListArray, writing offsets and carry indices.
# tooffsets[i] is the start of row i in the flat carry output.
# tocarry[tooffsets[i] + j] = fromstarts[i] + regular_start_i + j*step  for j in [0, count_i).
#
# Example: fromstarts=[0,3,5], fromstops=[3,5,6], start=1, stop=kSliceNone, step=1
#   row 0: length=3, rs=1, re=3 → count=2 → carry [1,2]
#   row 1: length=2, rs=1, re=2 → count=1 → carry [4]
#   row 2: length=1, rs=1, re=1 → count=0
#   tooffsets=[0,2,3,3], tocarry=[1,2,4]
def awkward_ListArray_getitem_next_range(
    tooffsets, tocarry, fromstarts, fromstops, lenstarts, start, stop, step
):
    if lenstarts == 0:
        tooffsets[:1] = 0
        return
    # create an int64 type specifically, so that _make_count_row() doesn't break
    counts = cp.empty(lenstarts, dtype=cp.int64)
    unary_transform(
        d_in=CountingIterator(fromstarts.dtype.type(0)),
        d_out=counts,
        op=_make_count_row(fromstarts, fromstops, start, stop, step, cp.int64),
        num_items=lenstarts,
    )

    offsets = cp.empty(lenstarts + 1, dtype=tooffsets.dtype)
    offsets[0] = 0
    inclusive_scan(
        d_in=counts,
        d_out=offsets[1:],
        op=OpKind.PLUS,
        init_value=None,
        num_items=lenstarts,
    )
    tooffsets[: lenstarts + 1] = offsets

    total = int(offsets[lenstarts])
    if total == 0:
        return

    dtype = tocarry.dtype.type

    def fill_carry(q):
        lo = 0
        hi = lenstarts - 1
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if offsets[mid] <= q:
                lo = mid
            else:
                hi = mid - 1
        i = lo
        j = q - offsets[i]
        length = fromstops[i] - fromstarts[i]
        rs = start
        if step > 0:
            if rs == _kSliceNone:
                rs = 0
            elif rs < 0:
                rs = rs + length
            if rs < 0:
                rs = 0
            if rs > length:
                rs = length
        else:
            if rs == _kSliceNone:
                rs = length - 1
            elif rs < 0:
                rs = rs + length
            if rs < -1:
                rs = -1
            if rs > length - 1:
                rs = length - 1
        return dtype(fromstarts[i] + rs + j * step)

    unary_transform(
        d_in=CountingIterator(dtype(0)),
        d_out=tocarry[:total],
        op=fill_carry,
        num_items=total,
    )


## NOT USED (revisit later) (error checks take too much time)
# Selects one element per row using two simultaneous advanced indices.
# fromadvanced[i] selects which index in fromarray to use for row i.
# tocarry[i] = fromstarts[i] + fromarray[fromadvanced[i]]  (content index)
# toadvanced[i] = i  (output advanced position)
# Raises ValueError if stops < starts, stops > lencontent, or index out of range.
#
# Example: fromstarts=[0,3], fromstops=[3,5], fromarray=[1,2,0], fromadvanced=[2,0]
#   row 0: reg_at=fromarray[2]=0 → tocarry[0]=0
#   row 1: reg_at=fromarray[0]=1 → tocarry[1]=4
#   toadvanced=[0,1]
def awkward_ListArray_getitem_next_array_advanced(
    tocarry,
    toadvanced,
    fromstarts,
    fromstops,
    fromarray,
    fromadvanced,
    lenstarts,
    lencontent,
):
    if lenstarts == 0:
        return

    starts = fromstarts[:lenstarts]
    stops = fromstops[:lenstarts]
    if cp.any(stops < starts):
        raise ValueError("stops[i] < starts[i]")
    nonempty = starts != stops
    if cp.any(nonempty & (stops > lencontent)):
        raise ValueError("stops[i] > len(content)")

    dtype = tocarry.dtype.type
    adtype = toadvanced.dtype.type

    def fill(i):
        length = fromstops[i] - fromstarts[i]
        reg_at = fromarray[fromadvanced[i]]
        if reg_at < 0:
            reg_at = reg_at + length
        if reg_at < 0 or reg_at >= length:
            tocarry[i] = dtype(-1)
        else:
            tocarry[i] = dtype(fromstarts[i] + reg_at)
        toadvanced[i] = adtype(i)
        return dtype(0)

    unary_transform(
        d_in=CountingIterator(dtype(0)),
        d_out=DiscardIterator(),
        op=fill,
        num_items=lenstarts,
    )
    if cp.any(tocarry[:lenstarts] < 0):
        raise ValueError("index out of range")


## NOT USED (revisit later) (error checks take too much time)
# Indexes each row of a ListArray by a 1-D array, producing lenstarts*lenarray outputs.
# tocarry[i*lenarray + j] = fromstarts[i] + fromarray[j]  (content index)
# toadvanced[i*lenarray + j] = j  (which fromarray entry was used)
# Raises ValueError if stops < starts, stops > lencontent, or index out of range.
#
# Example: fromstarts=[0,3], fromstops=[3,5], fromarray=[1,0], lenstarts=2, lenarray=2
#   tocarry=[1,0, 4,3],  toadvanced=[0,1, 0,1]
def awkward_ListArray_getitem_next_array(
    tocarry,
    toadvanced,
    fromstarts,
    fromstops,
    fromarray,
    lenstarts,
    lenarray,
    lencontent,
):
    if lenstarts == 0 or lenarray == 0:
        return

    starts = fromstarts[:lenstarts]
    stops = fromstops[:lenstarts]
    if cp.any(stops < starts):
        raise ValueError("stops[i] < starts[i]")
    nonempty = starts != stops
    if cp.any(nonempty & (stops > lencontent)):
        raise ValueError("stops[i] > len(content)")

    dtype = tocarry.dtype.type
    adtype = toadvanced.dtype.type

    def fill(k):
        i = k // lenarray
        j = k % lenarray
        length = fromstops[i] - fromstarts[i]
        reg_at = fromarray[j]
        if reg_at < 0:
            reg_at = reg_at + length
        if reg_at < 0 or reg_at >= length:
            tocarry[k] = dtype(-1)
        else:
            tocarry[k] = dtype(fromstarts[i] + reg_at)
        toadvanced[k] = adtype(j)
        return dtype(0)

    unary_transform(
        d_in=CountingIterator(dtype(0)),
        d_out=DiscardIterator(),
        op=fill,
        num_items=lenstarts * lenarray,
    )
    if cp.any(tocarry[: lenstarts * lenarray] < 0):
        raise ValueError("index out of range")


## NOT USED (revisit later) (error checks take too much time)
# Expands a ListArray to accommodate jagged indexing, producing length*jaggedsize outputs.
# Each row i must have exactly jaggedsize elements.
# multistarts[i*jaggedsize + j] = singleoffsets[j]
# multistops[i*jaggedsize + j]  = singleoffsets[j+1]
# tocarry[i*jaggedsize + j]     = fromstarts[i] + j
# Raises ValueError if stops < starts or row size != jaggedsize.
#
# Example: fromstarts=[0,2], fromstops=[2,4], singleoffsets=[0,3,5], jaggedsize=2, length=2
#   multistarts=[0,3, 0,3],  multistops=[3,5, 3,5],  tocarry=[0,1, 2,3]
def awkward_ListArray_getitem_jagged_expand(
    multistarts,
    multistops,
    singleoffsets,
    tocarry,
    fromstarts,
    fromstops,
    jaggedsize,
    length,
):
    if length == 0 or jaggedsize == 0:
        return

    starts = fromstarts[:length]
    stops = fromstops[:length]
    if cp.any(stops < starts):
        raise ValueError("stops[i] < starts[i]")
    if cp.any((stops - starts) != jaggedsize):
        raise ValueError("cannot fit jagged slice into nested list")

    dtype_ms = multistarts.dtype.type
    dtype_me = multistops.dtype.type
    dtype_c = tocarry.dtype.type

    def fill(k):
        i = k // jaggedsize
        j = k % jaggedsize
        multistarts[k] = dtype_ms(singleoffsets[j])
        multistops[k] = dtype_me(singleoffsets[j + 1])
        tocarry[k] = dtype_c(fromstarts[i] + j)
        return dtype_c(0)

    unary_transform(
        d_in=CountingIterator(dtype_c(0)),
        d_out=DiscardIterator(),
        op=fill,
        num_items=length * jaggedsize,
    )


# Copies fromstarts/fromstops into tostarts/tostops at the given offsets, adding base.
# tostarts[tostartsoffset + i] = fromstarts[i] + base
# tostops[tostopsoffset + i]   = fromstops[i] + base
#
# Example: fromstarts=[0,3], fromstops=[3,5], tostartsoffset=2, tostopsoffset=2, base=10
#   tostarts[2:4] = [10, 13],  tostops[2:4] = [13, 15]
def awkward_ListArray_fill(
    tostarts,
    tostartsoffset,
    tostops,
    tostopsoffset,
    fromstarts,
    fromstops,
    length,
    base,
):
    if length == 0:
        return
    # these two calls can be fused into one with unary_transform, but the performance is the same
    cp.add(
        fromstarts[:length],
        base,
        out=tostarts[tostartsoffset : tostartsoffset + length],
    )
    cp.add(
        fromstops[:length], base, out=tostops[tostopsoffset : tostopsoffset + length]
    )


# Builds an output index for rpad_and_clip on an IndexedOptionArray at axis=1.
# frommask[i] != 0 means the position is masked (option-None).
# toindex[i] = -1 if masked, else the 0-based position among non-masked elements up to i.
#
# Example: frommask=[0,1,0,0,1], length=5
#   valid counts: [1,1,2,3,3] → toindex=[0,-1,1,2,-1]
def awkward_IndexedOptionArray_rpad_and_clip_mask_axis1(toindex, frommask, length):
    if length == 0:
        return

    # valid[i]=1 where not masked, 0 where masked — reused for both scan and output.
    valid = (frommask[:length] == 0).astype(toindex.dtype, copy=False)
    inclusive_scan(
        d_in=valid,
        d_out=toindex[:length],
        op=OpKind.PLUS,
        init_value=None,
        num_items=length,
    )
    # Branchless in-place: valid=1 → toindex[i]-1, valid=0 → -1. No temporaries.
    toindex[:length] *= valid
    toindex[:length] -= 1


# Checks that every entry in index is a valid content index.
# If not isoption: raises ValueError when index[i] < 0.
# Always raises ValueError when index[i] >= lencontent.
#
# Example: index=[0,2,1], length=3, lencontent=3, isoption=False → no error
#          index=[0,3,1], length=3, lencontent=3, isoption=False → error at i=1
def awkward_IndexedArray_validity(index, length, lencontent, isoption):
    if length == 0:
        return
    idx = index[:length]
    if isoption:
        if cp.max(idx) >= lencontent:
            raise ValueError(
                "index[i] >= len(content) in compiled CUDA code (awkward_IndexedArray_validity)"
            )
    else:
        # Zero-copy view as unsigned: negative signed values bitcast to large
        # unsigned values, so one max catches both "x < 0" and "x >= lencontent"
        # with a single kernel instead of a min + max pair.
        udtype = cp.dtype(f"u{idx.dtype.itemsize}")
        if int(cp.max(idx.view(udtype))) >= lencontent:
            if cp.any(idx < 0):
                raise ValueError(
                    "index[i] < 0 in compiled CUDA code (awkward_IndexedArray_validity)"
                )
            raise ValueError(
                "index[i] >= len(content) in compiled CUDA code (awkward_IndexedArray_validity)"
            )


# Computes nextshifts for a non-local indexed reduction that starts from existing shifts.
# For each valid position (index[i] >= 0) at output slot k:
#   nextshifts[k] = shifts[i] + (number of null positions before i)
# Null positions (index[i] < 0) contribute to the null count but produce no output slot.
#
# Example: index=[0,-1,2,-1,3], shifts=[0,0,0,0,0], length=5
#   valid at i=0,2,4 → k=0,1,2; nullsum before each = 0,1,2
#   nextshifts = [0+0, 0+1, 0+2] = [0,1,2]
def awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64(
    nextshifts, index, length, shifts
):
    if length == 0:
        return

    # Dynamically select dtypes based on input arrays
    idx_dtype = index.dtype
    out_dtype = nextshifts.dtype

    # 1. Calculate the mapping to the output slots (k) using one scan
    # valid_mask is 1 if index >= 0, else 0
    valid_mask = (index[:length] >= 0).astype(idx_dtype, copy=False)
    scan_k = cp.empty(length, dtype=idx_dtype)

    inclusive_scan(
        d_in=valid_mask,
        d_out=scan_k,
        op=OpKind.PLUS,
        init_value=None,
        num_items=length,
    )

    def compute_and_scatter(i):
        idx_val = index[i]
        if idx_val >= 0:
            k = scan_k[i] - 1
            # Relationship: null_count = i - (valid_count_before_i)
            # which simplifies to i - k
            null_count = i - k
            nextshifts[k] = out_dtype.type(shifts[i] + null_count)
        return 0

    # We use a CountingIterator to drive the transform across the input indices
    unary_transform(
        d_in=cp.arange(length, dtype=idx_dtype),
        d_out=DiscardIterator(),
        op=compute_and_scatter,
        num_items=length,
    )


# Converts a starts array into an offsets array for an indexed reduction.
# outoffsets[i] = starts[i] for i in [0, startslength), outoffsets[startslength] = outindexlength.
#
# Example: starts=[0,3,5], startslength=3, outindexlength=7
#   outoffsets=[0,3,5,7]
def awkward_IndexedArray_reduce_next_fix_offsets_64(
    outoffsets, starts, startslength, outindexlength
):
    if startslength == 0:
        outoffsets[0] = outindexlength
        return
    if starts.dtype == outoffsets.dtype:
        outoffsets[:startslength] = starts[:startslength]
    else:
        # just in case
        cp.copyto(outoffsets[:startslength], starts[:startslength], casting="same_kind")
    outoffsets[startslength] = outindexlength


def _make_count_valid(index, fromstarts, fromstops, out_dtype):
    def count_valid(i):
        start = fromstarts[i]
        stop = fromstops[i]
        count = out_dtype.type(0)
        for j in range(start, stop):
            if index[j] >= 0:
                count = count + out_dtype.type(1)
        return count

    return count_valid


# For each row i in [0, length): counts non-negative entries in index[fromstarts[i]:fromstops[i]].
# tostarts[i] / tostops[i] are the prefix-sum start/end of those non-negative entries.
# tolength[0] = total number of non-negative entries across all rows.
#
# Example:
#   index      = [3, -1, 5,  2, -1,  7, -1, -1, 4]
#   fromstarts = [0, 3, 6],  fromstops = [3, 6, 9],  length = 3
#   row 0: [3, -1, 5]   → 2 valid
#   row 1: [2, -1,  7]  → 2 valid
#   row 2: [-1, -1, 4]  → 1 valid
#   tostarts = [0, 2, 4],  tostops = [2, 4, 5],  tolength = [5]
def awkward_IndexedArray_ranges_next_64(
    index, fromstarts, fromstops, length, tostarts, tostops, tolength
):
    if length == 0:
        tolength[:1] = 0
        return
    out_dtype = tostarts.dtype
    row_counts = cp.empty(length, dtype=out_dtype)
    unary_transform(
        d_in=CountingIterator(out_dtype.type(0)),
        d_out=row_counts,
        op=_make_count_valid(index, fromstarts[:length], fromstops[:length], out_dtype),
        num_items=length,
    )
    scan = cp.empty(length + 1, dtype=out_dtype)
    scan[0] = 0
    inclusive_scan(
        d_in=row_counts,
        d_out=scan[1:],
        op=OpKind.PLUS,
        init_value=None,
        num_items=length,
    )
    tostarts[:length] = scan[:length]
    tostops[:length] = scan[1:]
    tolength[:1] = scan[length]


# Collects non-negative values from index[fromstarts[i]:fromstops[i]] for each row i
# into tocarry, compactly (no gaps).
#
# Example:
#   index      = [3, -1, 5,  2, -1,  7, -1, -1, 4]
#   fromstarts = [0, 3, 6],  fromstops = [3, 6, 9],  length = 3
#   row 0: [3, -1, 5]   → valid: [3, 5]
#   row 1: [2, -1,  7]  → valid: [2, 7]
#   row 2: [-1, -1, 4]  → valid: [4]
#   tocarry = [3, 5, 2, 7, 4]
def awkward_IndexedArray_ranges_carry_next_64(
    index, fromstarts, fromstops, length, tocarry
):
    if length == 0:
        return
    out_dtype = tocarry.dtype
    row_counts = cp.empty(length, dtype=out_dtype)
    unary_transform(
        d_in=CountingIterator(out_dtype.type(0)),
        d_out=row_counts,
        op=_make_count_valid(index, fromstarts[:length], fromstops[:length], out_dtype),
        num_items=length,
    )
    scan = cp.empty(length + 1, dtype=out_dtype)
    scan[0] = 0
    inclusive_scan(
        d_in=row_counts,
        d_out=scan[1:],
        op=OpKind.PLUS,
        init_value=None,
        num_items=length,
    )
    if int(scan[length]) == 0:
        return

    def _make_fill_carry_valid(index, starts, stops, scan, tocarry):
        def fill_carry(i):
            start = starts[i]
            stop = stops[i]
            out_pos = scan[i]
            for j in range(start, stop):
                val = index[j]
                if val >= 0:
                    tocarry[out_pos] = val
                    out_pos = out_pos + scan.dtype.type(1)
            return scan.dtype.type(0)

        return fill_carry

    unary_transform(
        d_in=CountingIterator(out_dtype.type(0)),
        d_out=DiscardIterator(),
        op=_make_fill_carry_valid(
            index, fromstarts[:length], fromstops[:length], scan, tocarry
        ),
        num_items=length,
    )


# Fills toindex[0..lenindex] with [0, 1, ..., lenindex-1, -1].
# (The last entry is -1 to represent the None slot appended for unique.)
#
# Example:
#   lenindex = 4,  len(toindex) >= 5
#   toindex = [0, 1, 2, 3, -1]
def awkward_IndexedArray_numnull_unique_64(toindex, lenindex):
    unary_transform(
        d_in=CountingIterator(toindex.dtype.type(0)),
        d_out=toindex[:lenindex],
        op=OpKind.IDENTITY,
        num_items=lenindex,
    )
    toindex[lenindex] = toindex.dtype.type(-1)


# For each element in fromindex[0..lenindex): sets numnull[i]=1 if fromindex[i]<0, else 0.
# tolength[0] = total number of nulls.
#
# Example:
#   fromindex = [3, -1, 5, -1, -1, 2],  lenindex = 6
#   numnull   = [0,  1, 0,  1,  1, 0]
#   tolength  = [3]
def awkward_IndexedArray_numnull_parents(numnull, tolength, fromindex, lenindex):
    if lenindex == 0:
        tolength[:1] = 0
        return
    idx = fromindex[:lenindex]

    def _make_is_null(idx):
        def is_null(x):
            return idx.dtype.type(1) if x < 0 else idx.dtype.type(0)

        return is_null

    unary_transform(
        d_in=idx, d_out=numnull[:lenindex], op=_make_is_null(idx), num_items=lenindex
    )
    scan = cp.empty(lenindex, dtype=idx.dtype)
    inclusive_scan(
        d_in=numnull[:lenindex],
        d_out=scan,
        op=OpKind.PLUS,
        init_value=None,
        num_items=lenindex,
    )
    tolength[:1] = scan[lenindex - 1]


# For each i in [0, lenindex): if fromindex[i] < 0 → toindex[i] = -1,
# else tocarry[scan[i]-1] = fromindex[i] and toindex[i] = scan[i]-1.
# Raises ValueError if fromindex[i] >= lencontent.
#
# Example:
#   fromindex  = [3, -1, 5, 2, -1],  lenindex = 5,  lencontent = 10
#   is_valid   = [1,   0, 1, 1,   0]
#   scan       = [1,   1, 2, 3,   3]  (cumulative sum of is_valid)
#   i=0: pos=0 → tocarry[0]=3, toindex[0]=0
#   i=1: null  → toindex[1]=-1
#   i=2: pos=1 → tocarry[1]=5, toindex[2]=1
#   i=3: pos=2 → tocarry[2]=2, toindex[3]=2
#   i=4: null  → toindex[4]=-1
#   tocarry = [3, 5, 2],  toindex = [0, -1, 1, 2, -1]
def awkward_IndexedArray_getitem_nextcarry_outindex(
    tocarry, toindex, fromindex, lenindex, lencontent
):
    if lenindex == 0:
        return
    idx = fromindex[:lenindex]
    if cp.any(idx >= lencontent):
        raise ValueError(
            "index out of range in compiled CUDA code"
            " (awkward_IndexedArray_getitem_nextcarry_outindex)"
        )

    out_dtype = idx.dtype

    def is_valid(x):
        return out_dtype.type(1) if x >= 0 else out_dtype.type(0)

    scan = cp.empty(lenindex, dtype=out_dtype)
    inclusive_scan(
        d_in=TransformIterator(idx, is_valid),
        d_out=scan,
        op=OpKind.PLUS,
        init_value=None,
        num_items=lenindex,
    )

    def _make_fill_outindex(idx, scan, tocarry):
        def fill_outindex(i):
            v = idx[i]
            if v >= 0:
                pos = scan[i] - scan.dtype.type(1)
                tocarry[pos] = v
                return pos
            return idx.dtype.type(-1)

        return fill_outindex

    unary_transform(
        d_in=CountingIterator(idx.dtype.type(0)),
        d_out=toindex[:lenindex],
        op=_make_fill_outindex(idx, scan, tocarry),
        num_items=lenindex,
    )


# Copies non-negative fromindex entries into tocarry compactly.
# Raises ValueError if any fromindex[i] < 0 or >= lencontent.
#
# Example:
#   fromindex = [3, 5, 2, 7],  lenindex = 4,  lencontent = 10
#   tocarry   = [3, 5, 2, 7]
def awkward_IndexedArray_getitem_nextcarry(tocarry, fromindex, lenindex, lencontent):
    if lenindex == 0:
        return
    idx = fromindex[:lenindex]
    if cp.any((idx < 0) | (idx >= lencontent)):
        raise ValueError(
            "index out of range in compiled CUDA code"
            " (awkward_IndexedArray_getitem_nextcarry)"
        )
    tocarry[:lenindex] = idx.astype(tocarry.dtype, copy=False)


# For each i in [0, outindexlength): if outindex[i] < 0 → empty slice (size 0),
# else outoffsets[k] = offsets[outindex[i]+1] - offsets[outindex[i]].
# outoffsets[0] = offsets[0]; outoffsets[k] = cumulative sizes.
# Raises ValueError if outindex[i]+1 >= offsetslength.
#
# Example:
#   outindex  = [1, -1, 2, 0],  outindexlength = 4
#   offsets   = [0, 3, 3, 7, 10],  offsetslength = 5
#   sizes: [offsets[2]-offsets[1], 0, offsets[3]-offsets[2], offsets[1]-offsets[0]]
#        = [0, 0, 4, 3]
#   outoffsets = [0, 0, 0, 4, 7]
def awkward_IndexedArray_flatten_none2empty(
    outoffsets, outindex, outindexlength, offsets, offsetslength
):
    if outindexlength == 0:
        outoffsets[:1] = offsets[:1].astype(outoffsets.dtype, copy=False)
        return
    idx = outindex[:outindexlength]
    if cp.any((idx >= 0) & (idx + 1 >= offsetslength)):
        raise ValueError(
            "flattening offset out of range in compiled CUDA code"
            " (awkward_IndexedArray_flatten_none2empty)"
        )

    def compute_size(i):
        v = idx[i]
        if v < 0:
            return offsets.dtype.type(0)
        return offsets[v + 1] - offsets[v]

    base = int(offsets[0])
    outoffsets[0] = base
    unary_transform(
        d_in=CountingIterator(idx.dtype.type(0)),
        d_out=outoffsets[1 : outindexlength + 1],
        op=compute_size,
        num_items=outindexlength,
    )
    inclusive_scan(
        d_in=outoffsets[1 : outindexlength + 1],
        d_out=outoffsets[1 : outindexlength + 1],
        op=OpKind.PLUS,
        init_value=None,
        num_items=outindexlength,
    )
    cp.add(
        outoffsets[1 : outindexlength + 1], base, out=outoffsets[1 : outindexlength + 1]
    )


# Replaces each -1 in toindex with (n_non_null + rank_among_nulls).
# Non-(-1) entries are unchanged. Operates in-place on toindex[0..length).
#
# Example:
#   toindex   = [0, -1, 2, -1, -1, 1],  length = 6
#   is_none   = [0,  1, 0,  1,  1, 0]
#   scan_null = [0,  1, 1,  2,  3, 3]  (cumulative sum of is_none)
#   n_non_null = 6 - 3 = 3
#   nulls get indices: 3+1-1=3, 3+2-1=4, 3+3-1=5
#   toindex   = [0, 3, 2, 4, 5, 1]   (a permutation of 0..5)
def awkward_Index_nones_as_index(toindex, length):
    if length == 0:
        return
    idx = toindex[:length]

    out_dtype = idx.dtype

    def is_none(x):
        return out_dtype.type(1) if x == -1 else out_dtype.type(0)

    scan_null = cp.empty(length, dtype=out_dtype)
    inclusive_scan(
        d_in=TransformIterator(idx, is_none),
        d_out=scan_null,
        op=OpKind.PLUS,
        init_value=None,
        num_items=length,
    )
    n_non_null = length - int(scan_null[length - 1])
    toindex[:length] = cp.where(
        idx == -1,
        (n_non_null + scan_null - 1).astype(toindex.dtype, copy=False),
        idx,
    )


# For each i in [0, length): computes starts_out[i], stops_out[i], mask_out[i]
# based on whether index_in[i] >= 0 (present) or < 0 (missing).
# scan = prefix sum of (index_in >= 0).
# starts_out[i] = offsets_in[scan[i]], stops_out[i] = offsets_in[scan[i+1]]
# mask_out[i] = i if present, -1 if missing.
#
# Example:
#   index_in   = [0, -1,  1,  2, -1],  length = 5
#   offsets_in = [0,  3,  7, 10]        (3 present items → 3 intervals)
#   scan       = [0,  1,  1,  2,  3,  3]  (0 prepended; cumulative is_present)
#   i=0 (present): starts=offsets_in[0]=0,  stops=offsets_in[1]=3,  mask=0
#   i=1 (missing): starts=offsets_in[1]=3,  stops=offsets_in[1]=3,  mask=-1
#   i=2 (present): starts=offsets_in[1]=3,  stops=offsets_in[2]=7,  mask=2
#   i=3 (present): starts=offsets_in[2]=7,  stops=offsets_in[3]=10, mask=3
#   i=4 (missing): starts=offsets_in[3]=10, stops=offsets_in[3]=10, mask=-1
#   starts_out = [0, 3, 3,  7, 10],  stops_out = [3, 3, 7, 10, 10],  mask_out = [0, -1, 2, 3, -1]
def awkward_Content_getitem_next_missing_jagged_getmaskstartstop(
    index_in, offsets_in, mask_out, starts_out, stops_out, length
):
    if length == 0:
        return
    idx = index_in[:length]

    out_dtype = idx.dtype

    def is_present(x):
        return out_dtype.type(1) if x >= 0 else out_dtype.type(0)

    scan = cp.empty(length + 1, dtype=out_dtype)
    scan[0] = 0
    inclusive_scan(
        d_in=TransformIterator(idx, is_present),
        d_out=scan[1:],
        op=OpKind.PLUS,
        init_value=None,
        num_items=length,
    )

    def _make_fill_jagged(idx, scan, offsets_in, starts_out, stops_out):
        def fill_jagged(i):
            s = scan[i]
            starts_out[i] = offsets_in[s]
            stops_out[i] = offsets_in[scan[i + scan.dtype.type(1)]]
            if idx[i] < 0:
                return idx.dtype.type(-1)
            return idx.dtype.type(i)

        return fill_jagged

    unary_transform(
        d_in=CountingIterator(idx.dtype.type(0)),
        d_out=mask_out[:length],
        op=_make_fill_jagged(
            idx, scan, offsets_in, starts_out[:length], stops_out[:length]
        ),
        num_items=length,
    )


# For each i in [0, length): toindex[i] = i if (mask[i] != 0) == validwhen, else -1.
#
# Example:
#   mask = [0, 1, 0, 1, 0],  length = 5,  validwhen = True
#   toindex = [-1, 1, -1, 3, -1]
def awkward_ByteMaskedArray_toIndexedOptionArray(toindex, mask, length, validwhen):
    if length == 0:
        return
    msk = mask[:length]
    out_dtype = toindex.dtype

    def _make_fill(msk, validwhen, out_dtype):
        def fill(i):
            return (
                out_dtype.type(i) if (msk[i] != 0) == validwhen else out_dtype.type(-1)
            )

        return fill

    unary_transform(
        d_in=CountingIterator(out_dtype.type(0)),
        d_out=toindex[:length],
        op=_make_fill(msk, validwhen, out_dtype),
        num_items=length,
    )


# For each valid i (where (mask[i] != 0) == valid_when), sets
# nextshifts[k] = number of null entries before position i,
# where k is the 0-indexed rank among valid entries.
# null_count_before[i] = i - k  (since null + valid = total up to i).
#
# Example:
#   mask = [0, 1, 0, 1, 0, 1],  length = 6,  valid_when = True
#   valid at i=1 (k=0): nextshifts[0] = 1
#   valid at i=3 (k=1): nextshifts[1] = 2
#   valid at i=5 (k=2): nextshifts[2] = 3
def awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64(
    nextshifts, mask, length, valid_when
):
    if length == 0:
        return
    out_dtype = nextshifts.dtype
    msk = mask[:length]
    scan_k = cp.empty(length, dtype=out_dtype)

    def _make_is_valid(valid_when, out_dtype):
        def is_valid(x):
            return out_dtype.type((x != 0) == valid_when)

        return is_valid

    inclusive_scan(
        d_in=TransformIterator(msk, _make_is_valid(valid_when, out_dtype)),
        d_out=scan_k,
        op=OpKind.PLUS,
        init_value=None,
        num_items=length,
    )

    def _make_scatter_nextshifts(msk, scan_k, nextshifts, valid_when, out_dtype):
        def scatter(i):
            is_v = (msk[i] != 0) == valid_when
            if is_v:
                k = scan_k[i] - out_dtype.type(1)
                nextshifts[k] = out_dtype.type(i) - k
            return out_dtype.type(0)

        return scatter

    unary_transform(
        d_in=CountingIterator(out_dtype.type(0)),
        d_out=DiscardIterator(),
        op=_make_scatter_nextshifts(msk, scan_k, nextshifts, valid_when, out_dtype),
        num_items=length,
    )


# Like awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64 but adds shifts[i]:
# nextshifts[k] = shifts[i] + null_count_before_i.
#
# Example:
#   mask = [0, 1, 0, 1],  length = 4,  valid_when = True,  shifts = [10, 20, 30, 40]
#   valid at i=1 (k=0): nextshifts[0] = 20 + 1 = 21
#   valid at i=3 (k=1): nextshifts[1] = 40 + 2 = 42
def awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_fromshifts_64(
    nextshifts, mask, length, valid_when, shifts
):
    if length == 0:
        return
    out_dtype = nextshifts.dtype
    msk = mask[:length]
    scan_k = cp.empty(length, dtype=out_dtype)

    def _make_is_valid(valid_when, out_dtype):
        def is_valid(x):
            return out_dtype.type((x != 0) == valid_when)

        return is_valid

    inclusive_scan(
        d_in=TransformIterator(msk, _make_is_valid(valid_when, out_dtype)),
        d_out=scan_k,
        op=OpKind.PLUS,
        init_value=None,
        num_items=length,
    )

    def _make_scatter_nextshifts_fromshifts(
        msk, scan_k, nextshifts, valid_when, shifts, out_dtype
    ):
        def scatter(i):
            is_v = (msk[i] != 0) == valid_when
            if is_v:
                k = scan_k[i] - out_dtype.type(1)
                null_count = out_dtype.type(i) - k
                nextshifts[k] = out_dtype.type(shifts[i]) + null_count
            return out_dtype.type(0)

        return scatter

    unary_transform(
        d_in=CountingIterator(out_dtype.type(0)),
        d_out=DiscardIterator(),
        op=_make_scatter_nextshifts_fromshifts(
            msk, scan_k, nextshifts, valid_when, shifts, out_dtype
        ),
        num_items=length,
    )


# Overlays mymask onto theirmask: tomask[i] = theirmask[i] | mine[i],
# where mine[i] = 1 if mymask[i] is null (i.e. (mymask[i] != 0) != validwhen).
# A 1 bit in mine propagates "null" into the output regardless of theirmask.
#
# Example:
#   theirmask = [0, 1, 0, 0],  mymask = [1, 0, 0, 1],  validwhen = True,  length = 4
#   mine[0] = (1!=0)!=True = 0  (mymask says valid)
#   mine[1] = (0!=0)!=True = 1  (mymask says null)
#   mine[2] = (0!=0)!=True = 1  (mymask says null)
#   mine[3] = (1!=0)!=True = 0  (mymask says valid)
#   tomask = [0|0, 1|1, 0|1, 0|0] = [0, 1, 1, 0]
def awkward_ByteMaskedArray_overlay_mask(tomask, theirmask, mymask, length, validwhen):
    if length == 0:
        return
    out_dtype = tomask.dtype
    their = theirmask[:length]
    my = mymask[:length]

    def _make_overlay(their, my, validwhen, out_dtype):
        def overlay(i):
            mine = out_dtype.type((my[i] != 0) != validwhen)
            return their[i] | mine

        return overlay

    unary_transform(
        d_in=CountingIterator(cp.int64(0)),
        d_out=tomask[:length],
        op=_make_overlay(their, my, validwhen, out_dtype),
        num_items=length,
    )


# For each i in [0, length): if (mask[i] != 0) == validwhen → tocarry[k] = i,
# outindex[i] = k; else outindex[i] = -1. k is the compact rank among valid entries.
#
# Example:
#   mask = [0, 1, 0, 1, 1],  length = 5,  validwhen = True
#   scan = [0, 1, 1, 2, 3]  (cumulative valid count)
#   tocarry  = [1, 3, 4]
#   outindex = [-1, 0, -1, 1, 2]
def awkward_ByteMaskedArray_getitem_nextcarry_outindex(
    tocarry, outindex, mask, length, validwhen
):
    if length == 0:
        return
    msk = mask[:length]
    out_dtype = tocarry.dtype
    scan = cp.empty(length, dtype=out_dtype)

    def _make_is_valid(validwhen, out_dtype):
        def is_valid(x):
            return out_dtype.type((x != 0) == validwhen)

        return is_valid

    inclusive_scan(
        d_in=TransformIterator(msk, _make_is_valid(validwhen, out_dtype)),
        d_out=scan,
        op=OpKind.PLUS,
        init_value=None,
        num_items=length,
    )

    def _make_fill_carry_outindex(msk, scan, tocarry, validwhen, out_dtype):
        def fill(i):
            is_v = (msk[i] != 0) == validwhen
            if is_v:
                pos = scan[i] - out_dtype.type(1)
                tocarry[pos] = out_dtype.type(i)
                return pos
            return out_dtype.type(-1)

        return fill

    unary_transform(
        d_in=CountingIterator(out_dtype.type(0)),
        d_out=outindex[:length],
        op=_make_fill_carry_outindex(msk, scan, tocarry, validwhen, out_dtype),
        num_items=length,
    )


# Expands a packed bitmask (bitmasklength bytes) to an indexed option array of
# n = bitmasklength * 8 entries. toindex[p] = p if valid, -1 if missing.
#
# lsb_order=True:  bit j of byte i → entry i*8+j  (bit 0 = LSB).
# lsb_order=False: bit 7-j of byte i → entry i*8+j  (bit 7 = MSB first).
# Validity is determined by (bit == validwhen).
#
# Example (lsb_order=True, validwhen=True, bitmasklength=1, frombitmask=[0b00001010]):
#   bits (LSB first): 0,1,0,1,0,0,0,0
#   toindex = [-1, 1, -1, 3, -1, -1, -1, -1]
def awkward_BitMaskedArray_to_IndexedOptionArray(
    toindex, frombitmask, bitmasklength, validwhen, lsb_order
):
    if bitmasklength == 0:
        return
    n_elements = bitmasklength * 8
    out_dtype = toindex.dtype

    def _make_fill(frombitmask, validwhen, lsb_order, out_dtype):
        def fill(p):
            byte = out_dtype.type(frombitmask[p // 8])
            bit_idx = p % 8
            if lsb_order:
                bit = (byte >> bit_idx) & out_dtype.type(1)
            else:
                bit = (byte >> (7 - bit_idx)) & out_dtype.type(1)
            return p if (bit != out_dtype.type(0)) == validwhen else out_dtype.type(-1)

        return fill

    unary_transform(
        d_in=CountingIterator(out_dtype.type(0)),
        d_out=toindex[:n_elements],
        op=_make_fill(frombitmask, validwhen, lsb_order, out_dtype),
        num_items=n_elements,
    )


# Expands a packed bitmask (bitmasklength bytes) to a byte mask of
# n = bitmasklength * 8 entries. tobytemask[p] = 0 if valid, 1 if missing.
# (bit != validwhen) determines whether a position is missing.
#
# lsb_order=True:  bit j of byte i → entry i*8+j.
# lsb_order=False: bit 7-j of byte i → entry i*8+j.
#
# Example (lsb_order=True, validwhen=True, bitmasklength=1, frombitmask=[0b00001010]):
#   bits (LSB first): 0,1,0,1,0,0,0,0
#   tobytemask = [1, 0, 1, 0, 1, 1, 1, 1]
def awkward_BitMaskedArray_to_ByteMaskedArray(
    tobytemask, frombitmask, bitmasklength, validwhen, lsb_order
):
    if bitmasklength == 0:
        return
    n_elements = bitmasklength * 8
    out_dtype = tobytemask.dtype

    def _make_fill(frombitmask, validwhen, lsb_order, out_dtype):
        def fill(p):
            byte = frombitmask[p // 8]
            bit_idx = p % 8
            if lsb_order:
                bit = (byte >> bit_idx) & 1
            else:
                bit = (byte >> (7 - bit_idx)) & 1
            return out_dtype.type((bit != 0) != validwhen)

        return fill

    unary_transform(
        d_in=CountingIterator(cp.int64(0)),
        d_out=tobytemask[:n_elements],
        op=_make_fill(frombitmask, validwhen, lsb_order, out_dtype),
        num_items=n_elements,
    )


# Adjusts a sorted carry array in-place so that positions are expressed
# relative to the start of each outer bin (used after sorting with shifts).
#
# Phase 1 (bin-major): for each bin b, add fromoffsets[b] to every element
#   toptr[fromoffsets[b] .. fromoffsets[b+1])  (makes positions absolute)
# Phase 2 (per-shift): for each i in [0, length):
#   parent = fromparents[i];  start = fromstarts[parent]
#   toptr[i] += fromshifts[toptr[i]] - start
#   (applies shift correction and subtracts the bin's start to make relative)
#
# Example: outlength=2, fromoffsets=[0,3,5], toptr=[2,0,1, 1,0]
#   Phase 1: toptr[0:3] += 0 → [2,0,1]; toptr[3:5] += 3 → [4,3]  →  [2,0,1,4,3]
#   Phase 2 (length=5): fromshifts and fromstarts adjust each element
def awkward_NumpyArray_rearrange_shifted(
    toptr, fromshifts, length, fromoffsets, outlength, fromparents, fromstarts
):
    if outlength == 0:
        return
    total = int(fromoffsets[outlength])
    n = total if total > length else length
    if n == 0:
        return

    # Slice to stable shapes so the closure captures only CuPy arrays.
    # cuda.compute keys closures on (dtype, shape) for arrays — value equality,
    # stable across calls. Python int scalars use id() — unstable for large ints,
    # causing recompilation every call.
    # total   → fromoffsets_v[-1]         (one broadcast-cached global read)
    # outlength → fromoffsets_v.shape[0]-1 (Numba struct field, register-level)
    # length  → fromparents_v.shape[0]    (Numba struct field, register-level)
    fromoffsets_v = fromoffsets[: outlength + 1]
    fromparents_v = fromparents[:length]

    def rearrange(i):
        # Phase 1: convert local sorted position → absolute flat index
        total_val = fromoffsets_v[fromoffsets_v.shape[0] - np.int64(1)]
        if i < total_val:
            lo = np.int64(0)
            hi = fromoffsets_v.shape[0] - np.int64(1)
            while lo < hi:
                mid = lo + ((hi - lo + np.int64(1)) >> np.int64(1))
                if fromoffsets_v[mid] <= i:
                    lo = mid
                else:
                    hi = mid - np.int64(1)
            abs_pos = toptr[i] + fromoffsets_v[lo]
        else:
            abs_pos = toptr[i]
        # Phase 2: apply shift correction, subtract bin start → back to relative
        if i < fromparents_v.shape[0]:
            toptr[i] = abs_pos + fromshifts[abs_pos] - fromstarts[fromparents_v[i]]
        else:
            toptr[i] = abs_pos
        return np.int64(0)

    unary_transform(
        d_in=CountingIterator(cp.int64(0)),
        d_out=DiscardIterator(),
        op=rearrange,
        num_items=n,
    )


# For each bin b in [0, outlength) and each i in [offsets[b], offsets[b+1]):
#   if (mask[i] != 0) == validwhen (i.e. element is valid):
#     nextcarry[k] = i,  outindex[i] = k,  k++
#   else:
#     outindex[i] = -1
#   nextoffsets[b+1] = number of valid elements in [0, offsets[b+1])
#
# Example: mask=[1,0,1,1], offsets=[0,2,4], outlength=2, validwhen=True
#   valid=[T,F,T,T], cumvalid=[0,1,1,2,3]
#   nextcarry=[0,2,3], outindex=[0,-1,1,2], nextoffsets=[0,1,3]
def awkward_ByteMaskedArray_reduce_next_64(
    nextcarry, nextoffsets, outindex, mask, offsets, outlength, validwhen
):
    nextoffsets[0] = 0
    if outlength == 0:
        return
    total = int(offsets[outlength])
    if total == 0:
        nextoffsets[1 : outlength + 1] = 0
        return

    out_dtype = nextcarry.dtype
    valid = ((mask[:total] != 0) == validwhen).astype(out_dtype)
    scan = cp.empty(total, dtype=out_dtype)
    inclusive_scan(
        d_in=valid,
        d_out=scan,
        op=OpKind.PLUS,
        init_value=None,
        num_items=total,
    )

    def fill_outindex(i):
        if (mask[i] != 0) == validwhen:
            k = scan[i] - 1
            nextcarry[k] = out_dtype.type(i)
            return k
        return out_dtype.type(-1)

    unary_transform(
        d_in=CountingIterator(out_dtype.type(0)),
        d_out=outindex[:total],
        op=fill_outindex,
        num_items=total,
    )

    off_dtype = offsets.dtype.type

    def fill_nextoffsets(j):
        stop = offsets[j + 1]
        nextoffsets[j + 1] = out_dtype.type(0) if stop == 0 else scan[stop - 1]
        return off_dtype(0)

    unary_transform(
        d_in=CountingIterator(off_dtype(0)),
        d_out=DiscardIterator(),
        op=fill_nextoffsets,
        num_items=outlength,
    )


# Converts global flat indices from argmin/argmax into local within-list indices.
# After argmin/argmax, toptr[i] holds the flat index of the winning element across all
# lists. Subtracting starts[i] (the offset where list i begins) makes it relative
# to that list. Negative values (empty-list sentinels) are left untouched.
#
# Example: flat=[10,20,30,40,50], lists=[[10,20,30],[40,50]], starts=[0,3]
#   argmin reduction gives toptr=[0, 3]  (global flat indices)
#   after adjust:          toptr=[0, 0]  (local indices within each list)
def awkward_NumpyArray_reduce_adjust_starts_64(toptr, outlength, offsets, starts):
    if outlength == 0:
        return

    out_dtype = toptr.dtype.type

    def adjust(i):
        v = toptr[i]
        if v >= 0:
            toptr[i] = v - starts[i]
        return out_dtype(0)

    unary_transform(
        d_in=CountingIterator(out_dtype(0)),
        d_out=DiscardIterator(),
        op=adjust,
        num_items=outlength,
    )


# Builds nextcarry/nextoffsets for a nonlocal reduction over a RegularArray.
# Iterates bin-major then column-major: for each bin b in [0, outlength) and
# column j in [0, size), one entry per row i in [offsets[b], offsets[b+1]):
#   nextcarry[k] = i * size + j,  nextoffsets[b*size+j+1] = k+1
#
# Example: offsets=[0,2,4], size=2, length=4, outlength=2
#   Input — a ListOffsetArray of a RegularArray (2 bins, 2 columns per row):
#     [              # bin 0 (rows 0-1)
#       [a, b],      # row 0
#       [c, d],      # row 1
#     ],
#     [              # bin 1 (rows 2-3)
#       [e, f],      # row 2
#       [g, h],      # row 3
#     ]
#   Goal — reduce within each bin, column-by-column:
#     [[a+c, b+d],   # bin 0
#      [e+g, f+h]]   # bin 1
#   Output:
#     nextcarry   = [0,2, 1,3, 4,6, 5,7]  (flat indices, grouped by (bin,col))
#     nextoffsets = [0, 2, 4, 6, 8]
def awkward_RegularArray_reduce_nonlocal_preparenext_64(
    nextcarry, nextoffsets, offsets, size, length, outlength
):
    nextoffsets[0] = 0
    if outlength == 0 or size == 0:
        return
    per_bin_counts = (offsets[1 : outlength + 1] - offsets[:outlength]).astype(
        nextoffsets.dtype
    )
    per_nextbin_counts = cp.repeat(per_bin_counts, size)
    cp.cumsum(per_nextbin_counts, out=nextoffsets[1 : outlength * size + 1])
    if length == 0:
        return
    total_out = int(nextoffsets[outlength * size])
    if total_out == 0:
        return
    n_nextbins = outlength * size
    # Only CuPy arrays in the closure → cache keyed by (dtype, shape)
    # nextoffsets_v is kept for its shape (gives sz = n_nextbins // outlength);
    # its data is never read inside fill_nextcarry.
    nextoffsets_v = nextoffsets[:n_nextbins]
    offsets_v = offsets[: outlength + 1].astype(nextcarry.dtype)

    def fill_nextcarry(k):
        # sz and outlength from array shapes — register-level, no GPU reads.
        sz = nextoffsets_v.shape[0] // (offsets_v.shape[0] - np.int64(1))
        # Binary search over offsets_v (outlength elements)
        # Find largest b such that offsets_v[b]*sz <= k.
        lo = np.int64(0)
        hi = offsets_v.shape[0] - np.int64(2)
        while lo < hi:
            mid = lo + ((hi - lo + np.int64(1)) >> np.int64(1))
            if offsets_v[mid] * sz <= k:
                lo = mid
            else:
                hi = mid - np.int64(1)
        b = lo
        count_b = offsets_v[b + np.int64(1)] - offsets_v[b]
        off = k - offsets_v[b] * sz
        j = off // count_b
        row_in_bin = off - j * count_b
        return (offsets_v[b] + row_in_bin) * sz + j

    unary_transform(
        d_in=CountingIterator(np.int64(0)),
        d_out=nextcarry[:total_out],
        op=fill_nextcarry,
        num_items=total_out,
    )


# Copies fromindex[0:length] into toindex at toindexoffset, adding base to
# non-negative values and mapping negatives to -1.
# toindex[toindexoffset + i] = fromindex[i] < 0 ? -1 : fromindex[i] + base
#
# Example: fromindex=[1,-1,3], toindexoffset=2, base=10
#   toindex[2:5] = [11, -1, 13]
def awkward_IndexedArray_fill(toindex, toindexoffset, fromindex, length, base):
    if length == 0:
        return

    out_dtype = toindex.dtype.type

    def fill(i):
        x = fromindex[i]
        return out_dtype(-1) if x < 0 else out_dtype(x + base)

    unary_transform(
        d_in=CountingIterator(out_dtype(0)),
        d_out=toindex[toindexoffset : toindexoffset + length],
        op=fill,
        num_items=length,
    )


# Copies `length` values from fromindex into toindex starting at toindexoffset,
# casting to toindex's dtype.
#
# Example: fromindex=[10, 20, 30], toindexoffset=2, length=3
#   toindex[2:5] = [10, 20, 30]
def awkward_UnionArray_fillindex(toindex, toindexoffset, fromindex, length):
    if length == 0:
        return

    def fill_idx(i):
        return toindex.dtype.type(fromindex[i])

    unary_transform(
        d_in=CountingIterator(fromindex.dtype.type(0)),
        d_out=toindex[toindexoffset : toindexoffset + length],
        op=fill_idx,
        num_items=length,
    )


# Fills toindex[toindexoffset + i] = i for i in [0, length).
#
# Example: toindexoffset=3, length=4
#   toindex[3:7] = [0, 1, 2, 3]
def awkward_UnionArray_fillindex_count(toindex, toindexoffset, length):
    if length == 0:
        return
    unary_transform(
        d_in=CountingIterator(toindex.dtype.type(0)),
        d_out=toindex[toindexoffset : toindexoffset + length],
        op=lambda i: i,
        num_items=length,
    )


# Copies fromindex to toindex, replacing negative values with 0.
# toindex[i] = fromindex[i] if fromindex[i] >= 0 else 0
#
# Example: fromindex=[-1, 3, -2, 7], length=4
#   toindex = [0, 3, 0, 7]
def awkward_UnionArray_fillna(toindex, fromindex, length):
    if length == 0:
        return

    def fill(x):
        return x if x >= toindex.dtype.type(0) else toindex.dtype.type(0)

    unary_transform(
        d_in=fromindex[:length],
        d_out=toindex[:length],
        op=fill,
        num_items=length,
    )


# Copies fromtags to totags at totagsoffset, adding base to each value.
# totags[totagsoffset + i] = fromtags[i] + base
#
# Example: fromtags=[0, 1, 0], totagsoffset=2, base=5, length=3
#   totags[2:5] = [5, 6, 5]
def awkward_UnionArray_filltags(totags, totagsoffset, fromtags, length, base):
    if length == 0:
        return

    def fill_tag(i):
        return totags.dtype.type(fromtags[i] + base)

    unary_transform(
        d_in=CountingIterator(fromtags.dtype.type(0)),
        d_out=totags[totagsoffset : totagsoffset + length],
        op=fill_tag,
        num_items=length,
    )


# Fills totags[totagsoffset + i] = base for i in [0, length).
#
# Example: totagsoffset=1, length=3, base=2
#   totags[1:4] = [2, 2, 2]
def awkward_UnionArray_filltags_const(totags, totagsoffset, length, base):
    if length == 0:
        return

    def fill_const(i):
        return totags.dtype.type(base)

    unary_transform(
        d_in=CountingIterator(totags.dtype.type(0)),
        d_out=totags[totagsoffset : totagsoffset + length],
        op=fill_const,
        num_items=length,
    )


# THIS KERNEL IS NOT USED (just for archive)
# Computes the total number of output elements when flattening a UnionArray.
# For each element i: adds offsetsraws[fromtags[i]][fromindex[i]+1]
#                              - offsetsraws[fromtags[i]][fromindex[i]]
# to the total.  offsetsraws is a Python list of per-content offset GPU arrays.
# Stores the result in total_length[0].
#
# Example: fromtags=[0,1,0], fromindex=[0,0,1], length=3
#   offsetsraws[0]=[0,3,5], offsetsraws[1]=[0,2]
#   sizes: [3-0, 2-0, 5-3] = [3, 2, 2] → total_length[0] = 7
def awkward_UnionArray_flatten_length(
    total_length, fromtags, fromindex, length, offsetsraws
):
    if length == 0:
        total_length[0] = 0
        return
    tags = fromtags[:length]
    idxs = fromindex[:length]

    type_starts = cp.array(
        [0]
        + [sum(len(o) for o in offsetsraws[: k + 1]) for k in range(len(offsetsraws))],
        dtype=cp.int64,
    )
    all_offsets = cp.concatenate(offsetsraws)

    flat_idx = type_starts[tags] + idxs
    sizes = all_offsets[flat_idx + 1] - all_offsets[flat_idx]
    total_length[0] = int(cp.sum(sizes))


# THIS KERNEL IS NOT USED (just for archive)
# Flattens a UnionArray, combining all sublists into a flat output.
# Writes tooffsets (length+1), totags (total_length), and toindex (total_length).
# offsetsraws is a Python list of per-content offset GPU arrays.
#
# For each element i:
#   tag = fromtags[i], idx = fromindex[i]
#   start = offsetsraws[tag][idx], stop = offsetsraws[tag][idx+1]
#   tooffsets[i+1] = tooffsets[i] + (stop - start)
#   for j in [start, stop): totags[k] = tag, toindex[k] = j
#
# Example: fromtags=[0,1], fromindex=[0,0], offsetsraws[0]=[0,3], offsetsraws[1]=[0,2]
#   tooffsets=[0,3,5], totags=[0,0,0,1,1], toindex=[0,1,2,0,1]
def awkward_UnionArray_flatten_combine(
    totags, toindex, tooffsets, fromtags, fromindex, length, offsetsraws
):
    tooffsets[0] = 0
    if length == 0:
        return
    if len(offsetsraws) == 2:
        offsets0, offsets1 = offsetsraws[0], offsetsraws[1]

        # Phase 1: compute per-element sublist sizes → inclusive scan → tooffsets
        sizes = cp.empty(length, dtype=tooffsets.dtype)

        def compute_size(i):
            j = fromindex.dtype.type(fromindex[i])
            if fromtags[i] == fromtags.dtype.type(0):
                return offsets0.dtype.type(offsets0[j + 1] - offsets0[j])
            return offsets1.dtype.type(offsets1[j + 1] - offsets1[j])

        unary_transform(
            d_in=CountingIterator(fromindex.dtype.type(0)),
            d_out=sizes,
            op=compute_size,
            num_items=length,
        )
        inclusive_scan(
            d_in=sizes,
            d_out=tooffsets[1 : length + 1],
            op=OpKind.PLUS,
            init_value=None,
            num_items=length,
        )

        total_len = int(tooffsets[length])
        if total_len == 0:
            return

        # Phase 2: binary-search each output position back to its parent row,
        # then write totags/toindex — no flat_pos or parent_ids temporaries
        def fill_out(q):
            lo = tooffsets.dtype.type(0)
            hi = tooffsets.dtype.type(length - 1)
            while lo < hi:
                mid = (lo + hi + tooffsets.dtype.type(1)) >> tooffsets.dtype.type(1)
                if tooffsets[mid] <= q:
                    lo = mid
                else:
                    hi = mid - tooffsets.dtype.type(1)
            i = lo
            j = fromindex.dtype.type(fromindex[i])
            local_offset = q - tooffsets[i]
            totags[q] = fromtags[i]
            if fromtags[i] == fromtags.dtype.type(0):
                toindex[q] = offsets0.dtype.type(offsets0[j]) + local_offset
            else:
                toindex[q] = offsets1.dtype.type(offsets1[j]) + local_offset
            return tooffsets.dtype.type(0)

        unary_transform(
            d_in=CountingIterator(tooffsets.dtype.type(0)),
            d_out=DiscardIterator(),
            op=fill_out,
            num_items=total_len,
        )
    else:
        tags = fromtags[:length]
        idxs = fromindex[:length]
        sizes = cp.zeros(length, dtype=tooffsets.dtype)
        global_start = cp.zeros(length, dtype=tooffsets.dtype)
        for k, offsets_k in enumerate(offsetsraws):
            pos = cp.where(tags == k)[0]
            if pos.size > 0:
                local_idxs = idxs[pos]
                starts = offsets_k[local_idxs]
                sizes[pos] = offsets_k[local_idxs + 1] - starts
                global_start[pos] = starts
        inclusive_scan(
            d_in=sizes,
            d_out=tooffsets[1 : length + 1],
            op=OpKind.PLUS,
            init_value=None,
            num_items=length,
        )
        total_len = int(tooffsets[length])
        if total_len == 0:
            return
        flat_pos = cp.arange(total_len, dtype=tooffsets.dtype)
        parent_ids = (
            cp.searchsorted(tooffsets[: length + 1], flat_pos, side="right") - 1
        )
        totags[:total_len] = tags[parent_ids].astype(totags.dtype, copy=False)
        toindex[:total_len] = (
            global_start[parent_ids] + flat_pos - tooffsets[parent_ids]
        ).astype(toindex.dtype, copy=False)


# Filters fromindex by fromtags == which, scattering matching entries compactly
# into tocarry, and writing the count into lenout[0].
#
# Example: fromtags=[0,1,0,1,0], fromindex=[10,20,30,40,50], length=5, which=0
#   tocarry = [10, 30, 50],  lenout[0] = 3
def awkward_UnionArray_project(lenout, tocarry, fromtags, fromindex, length, which):
    if length == 0:
        lenout[0] = 0
        return
    positions = cp.where(fromtags[:length] == which)[0]
    count = positions.size
    lenout[0] = count
    if count > 0:
        tocarry[:count] = fromindex[:length][positions].astype(
            tocarry.dtype, copy=False
        )


# This implementation uses a for loop.
# Since the size of Union array is usually small, I hope this implementation is performant enough.
#
# For each element i,
# toindex[i] = how many elements with the same tag appeared before position i
# current[k] = total count of elements with tag k.
#
# Example: fromtags=[0,1,0,2,1], size=3, length=5
#   Imagine you have a mixed list of apples and oranges, tagged by type:
#   tags: [ apple(0),  orange(1),  apple(0),  banana(2),  orange(1)]
#   Then for each item, its position within its own type will be:
#   toindex:  [  0,      0,       1,      0,       2   ]
#              1st      1st      2nd     1st      2nd
#              apple   orange   apple   banana   orange
# toindex = [0, 0, 1, 0, 1]
# current = [2, 2, 1]
def awkward_UnionArray_regular_index(toindex, current, size, fromtags, length):
    current[:] = current.dtype.type(0)
    if length == 0:
        return
    tags = fromtags[:length]
    for k in range(size):
        positions = cp.where(tags == k)[0]
        n = positions.size
        if n > 0:
            toindex[positions] = cp.arange(n, dtype=toindex.dtype)
        current[k] = toindex.dtype.type(n)


# Computes size = max(fromtags[0..length-1]) + 1, i.e. the number of distinct
# tag values needed to index into a per-content array.
#
# Example: fromtags=[0,2,1,2], length=4
#   size[0] = 3
def awkward_UnionArray_regular_index_getsize(size, fromtags, length):
    if length == 0:
        size[0] = 1
        return
    size[0] = cp.maximum(cp.max(fromtags[:length]) + 1, 1)


# Merges an outer UnionArray (outertags, outerindex) with an inner one
# (innertags, innerindex) for a specific (outerwhich, innerwhich) pair.
# Where outertags[i] == outerwhich and innertags[outerindex[i]] == innerwhich,
# writes totags[i] = towhich and toindex[i] = innerindex[outerindex[i]] + base.
# Other positions in totags/toindex are left unchanged.
#
# Example: outertags=[1,0,1], outerindex=[0,0,1], outerwhich=1
#          innertags=[0,1], innerindex=[5,7], innerwhich=1, towhich=2, base=10
#   i=0: outertags[0]=1==outerwhich → j=0, innertags[0]=0≠innerwhich → skip
#   i=2: outertags[2]=1==outerwhich → j=1, innertags[1]=1==innerwhich
#     totags[2]=2, toindex[2]=7+10=17
def awkward_UnionArray_simplify(
    totags,
    toindex,
    outertags,
    outerindex,
    innertags,
    innerindex,
    towhich,
    innerwhich,
    outerwhich,
    length,
    base,
):
    if length == 0:
        return

    def transform(i):
        if outertags[i] == outertags.dtype.type(outerwhich):
            j = outerindex.dtype.type(outerindex[i])
            if innertags[j] == innertags.dtype.type(innerwhich):
                totags[i] = towhich
                toindex[i] = innerindex[j] + base
        return outerindex.dtype.type(0)

    unary_transform(
        d_in=CountingIterator(outerindex.dtype.type(0)),
        d_out=DiscardIterator(),
        op=transform,
        num_items=length,
    )


# Fills toindex[toindexoffset : toindexoffset+length] with base, base+1, ..., base+length-1.
#
# Example: toindexoffset=3, length=4, base=10
#   toindex[3:7] = [10, 11, 12, 13]
def awkward_IndexedArray_fill_count(toindex, toindexoffset, length, base):
    if length == 0:
        return
    dtype = toindex.dtype.type
    # CountingIterator(dtype(base)) generates base, base+1, ...; _make_widening_cast
    # gives a stable cached identity-cast op so cuda.compute reuses the compiled kernel.
    unary_transform(
        d_in=CountingIterator(dtype(base)),
        d_out=toindex[toindexoffset : toindexoffset + length],
        op=_make_widening_cast(dtype, dtype),
        num_items=length,
    )


# Finds the within-list position of each null in an IndexedOptionArray.
# fromindex uses negative values to represent None; offsets/starts define bin
# boundaries. For each null, binary-searches offsets to find its bin, then
# subtracts starts[bin] to convert the flat position to a local index.
# Answers: "at what position within each list does a null appear?"
# Used to propagate null positions through reductions on option-typed arrays.
#
# Example: fromindex=[0,-1,2,-1,4], offsets=[0,3,5], starts=[0,3]
#   nulls at flat positions 1 and 3
#   pos 1 → bin 0 → local = 1 - 0 = 1
#   pos 3 → bin 1 → local = 3 - 3 = 0
#   toindex = [1, 0]
def awkward_IndexedArray_index_of_nulls(toindex, fromindex, offsets, outlength, starts):
    if outlength == 0:
        return
    total = int(offsets[outlength])
    if total == 0:
        return
    null_mask = fromindex[:total] < 0
    n_nulls = int(cp.sum(null_mask))
    if n_nulls == 0:
        return
    null_positions = cp.where(null_mask)[0]
    bin_ids = cp.searchsorted(offsets[1 : outlength + 1], null_positions, side="right")
    toindex[:n_nulls] = (null_positions - starts[bin_ids]).astype(
        toindex.dtype, copy=False
    )


# Maps each outer element to its paired inner position, or -1 if no inner
# element remains for that row. For each bin b and outer element i in
# [offsets[b], offsets[b+1]), the j-th element (j = i - offsets[b]) gets:
#   tocarry[i] = nextoffsets[b] + j  if nextoffsets[b] + j < nextoffsets[b+1]
#   tocarry[i] = -1                   otherwise
# (starts is accepted for API compatibility but unused)
#
# Example: offsets=[0,3,5], nextoffsets=[0,2,4], outlength=2
#   bin 0: i=0→j=0: inner=0+0=0 <2 ✓; i=1→j=1: inner=0+1=1 <2 ✓; i=2→j=2: 0+2=2 ≥2 →-1
#   bin 1: i=3→j=0: inner=2+0=2 <4 ✓; i=4→j=1: inner=2+1=3 <4 ✓
#   tocarry = [0, 1, -1, 2, 3]
def awkward_IndexedArray_local_preparenext_64(
    tocarry, starts, offsets, nextoffsets, outlength
):
    if outlength == 0:
        return
    total = int(offsets[outlength])
    if total == 0:
        return

    n_iters = outlength.bit_length()

    def fill(i):
        lo = 0
        hi = outlength
        for _ in range(n_iters):
            mid = (lo + hi) >> 1
            if offsets[mid + 1] <= i:
                lo = mid + 1
            else:
                hi = mid
        inner_pos = nextoffsets[lo] + i - offsets[lo]
        return inner_pos if inner_pos < nextoffsets[lo + 1] else -1

    unary_transform(
        d_in=CountingIterator(cp.int64(0)),
        d_out=tocarry[:total],
        op=fill,
        num_items=total,
    )


# Validates a UnionArray: checks tags[i] >= 0, index[i] >= 0,
# tags[i] < numcontents, and index[i] < lencontents[tags[i]] for all i.
# lencontents is a GPU array of length numcontents.
# Raises ValueError at the first failing element.
#
# Example: tags=[0,1], index=[2,5], numcontents=2, lencontents=[3,4]
#   index[1]=5 >= lencontents[1]=4 → raises ValueError
def awkward_UnionArray_validity(tags, index, length, numcontents, lencontents):
    if length == 0:
        return
    # operations on bools are not supported now inside numba closures
    err_flags = cp.empty(length, dtype=cp.int8)

    def check(i):
        ti = tags.dtype.type(tags[i])
        idxi = index.dtype.type(index[i])
        if ti < tags.dtype.type(0) or ti >= tags.dtype.type(numcontents):
            return cp.int8(1)
        if idxi < index.dtype.type(0) or idxi >= index.dtype.type(lencontents[ti]):
            return cp.int8(1)
        return cp.int8(0)

    unary_transform(
        d_in=CountingIterator(index.dtype.type(0)),
        d_out=err_flags,
        op=check,
        num_items=length,
    )
    any_err = cp.empty(1, dtype=cp.int8)
    reduce_into(
        d_in=err_flags,
        d_out=any_err,
        op=OpKind.MAXIMUM,
        h_init=np.zeros(1, dtype=np.int8),
        num_items=length,
    )
    if any_err[0] == 0:
        return
    # Error path: find the first failing element for error message
    lc = cp.asarray(lencontents[:numcontents], dtype=cp.int64)
    t = tags[:length].astype(cp.int64)
    idx = index[:length].astype(cp.int64, copy=False)
    bad_tag = (t < 0) | (t >= numcontents)
    if cp.any(bad_tag):
        i = int(cp.argmax(bad_tag))
        if int(t[i]) < 0:
            raise ValueError(
                "tags[i] < 0 in compiled CUDA code (awkward_UnionArray_validity)"
            )
        raise ValueError(
            "tags[i] >= len(contents) in compiled CUDA code (awkward_UnionArray_validity)"
        )
    bad_idx = (idx < 0) | (idx >= lc[t])
    i = int(cp.argmax(bad_idx))
    if int(idx[i]) < 0:
        raise ValueError(
            "index[i] < 0 in compiled CUDA code (awkward_UnionArray_validity)"
        )
    raise ValueError(
        "index[i] >= len(content[tags[i]]) in compiled CUDA code (awkward_UnionArray_validity)"
    )
