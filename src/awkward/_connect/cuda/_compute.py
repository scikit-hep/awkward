# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from cuda.compute import (
    CountingIterator,
    OpKind,
    segmented_reduce,
    unary_transform,
)

from awkward._connect import cuda
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy

cupy_nplike = Cupy.instance()
cp = cupy_nplike._module

numpy_nplike = Numpy.instance()
np = numpy_nplike._module

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


def segmented_sort(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable,
):
    from cuda.compute import SortOrder, segmented_sort

    cupy_nplike = Cupy.instance()
    cp = cupy_nplike._module

    # Ensure offsets are int64 as expected by segmented_sort
    if offsets.dtype != cp.int64:
        offsets = offsets.astype(cp.int64)

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
    d_input = input_data.astype(result.dtype, copy=False)
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
    d_in = input_data.view(cp.int8) if input_data.dtype == cp.bool_ else input_data

    d_out = result.view(cp.int8) if result.dtype == cp.bool_ else result

    start_o, end_o = make_segment_views(offsets_data)

    h_init = np.asarray(0, dtype=cp.int8)

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

    mapped_data = cp.empty(input_complex.shape, dtype=cp.int8)

    def is_nonzero_complex(c):
        # A complex number is non-zero if either real or imag is non-zero
        if c.real != 0 or c.imag != 0:
            return 1
        return 0

    unary_transform(
        d_in=input_complex,
        d_out=mapped_data,
        op=is_nonzero_complex,
        num_items=input_complex.size,
    )

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
    d_input = input_data.astype(result.dtype, copy=False)
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
    d_input = input_data.astype(result.dtype, copy=False)
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
    d_input = input_data.astype(result.dtype, copy=False)

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
    # Temporary workaround:
    # bool reductions currently fail on the numba side, so reinterpret as int8.
    if input_data.dtype == cp.bool_:
        input_data = input_data.view(cp.int8)

    start_o, end_o = make_segment_views(offsets_data)

    h_init = np.asarray(1, dtype=cp.int8)

    def prod_op(a, b):
        return a * b

    segmented_reduce(
        d_in=input_data,
        d_out=result,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=prod_op,
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

    mapped_input = cp.empty(input_complex.shape, dtype=cp.int8)

    def is_nonzero(c):
        return cp.int8(c.real != 0 or c.imag != 0)

    unary_transform(
        d_in=input_complex,
        d_out=mapped_input,
        op=is_nonzero,
        num_items=input_complex.size,
    )

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

    mapped_data = cp.empty(input_data.shape, dtype=result.dtype)

    def is_nonzero_map(x):
        return result.dtype.type(1) if x != 0 else result.dtype.type(0)

    unary_transform(
        d_in=input_data,
        d_out=mapped_data,
        op=is_nonzero_map,
        num_items=input_data.size,
    )

    start_o, end_o = make_segment_views(offsets_data)
    h_init = np.asarray(0, dtype=result.dtype)

    segmented_reduce(
        d_in=mapped_data,
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

    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_countnonzero(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]

        count = 0

        for i in range(start_idx, end_idx):
            if input_complex[i] != complex_dtype(0):
                count += 1

        return count

    segment_ids = CountingIterator(index_dtype(0))

    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_countnonzero,
        num_items=outlength,
    )


def awkward_index_rpad_and_clip_axis0(toindex, target, length):
    """
    Fill ``toindex[0..target)`` with the identity mapping ``[0..shorter)``
    followed by ``target - shorter`` entries of ``-1``, where
    ``shorter = min(target, length)``.

    Called from ``Content._pad_none_axis0`` in
    ``src/awkward/contents/content.py``.
    """
    dtype = toindex.dtype.type
    shorter = min(target, length)

    def fill(i):
        return dtype(i) if i < shorter else dtype(-1)

    counters = CountingIterator(dtype(0))
    unary_transform(
        d_in=counters,
        d_out=toindex,
        op=fill,
        num_items=target,
    )


def awkward_index_rpad_and_clip_axis1(tostarts, tostops, target, length):
    """
    Fills `tostarts` and `tostops` with rpad/clip offsets for axis=1 lists.
    Each list is padded or clipped to length `target`.
    """

    def fill(i):
        start = i * target
        end = start + target

        tostarts[i] = tostarts.dtype.type(start)
        return tostarts.dtype.type(end)

    segment_ids = CountingIterator(tostarts.dtype.type(0))

    unary_transform(
        d_in=segment_ids,
        d_out=tostops,
        op=fill,
        num_items=length,
    )


def awkward_missing_repeat(
    outindex,
    index,
    indexlength,
    repetitions,
    regularsize,
):
    """
    Repeats an index array `repetitions` times, adjusting valid (non-negative)
    indices by an offset of `regularsize` per repetition.
    Missing values (-1) are preserved.
    """
    index_dtype = outindex.dtype.type
    output_size = repetitions * indexlength

    reg_size = index_dtype(regularsize)
    idx_len = index_dtype(indexlength)

    def fill(counter):
        # Position in the original index array
        j = counter % idx_len
        # Which repetition block are we in?
        i = counter // idx_len

        base = index[j]

        # Awkward convention: -1 and lower are masked/missing
        if base >= 0:
            return index_dtype(base + i * reg_size)
        else:
            # Preserve the exact missing value (usually -1)
            return base

    counters = CountingIterator(index_dtype(0))

    unary_transform(
        d_in=counters,
        d_out=outindex,
        num_items=output_size,
        op=fill,
    )
