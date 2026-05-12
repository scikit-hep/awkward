# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from cuda.compute import (
    CountingIterator,
    reduce_into,
    unary_transform,
)

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


def as_complex_view(data):
    return data.view(infer_complex_dtype(data.dtype))


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


def parents_to_offsets(parents, parents_length):
    if parents_length == 0:
        return cp.array([0], dtype=parents.dtype)

    # count how many elements belong to each parent
    counts = cp.bincount(parents)

    # add a start offset
    offsets = cp.concatenate([cp.array([0], dtype=counts.dtype), counts.cumsum()])

    return offsets


def local_idx_from_parents(parents, parents_length):
    if parents_length == 0:
        return cp.empty(0, dtype=parents.dtype)

    # mark the beginning of each subarray
    new_group = cp.empty(parents_length, dtype=cp.bool_)
    new_group[0] = True
    new_group[1:] = parents[1:] != parents[:-1]

    # find the start index of each subarray
    group_starts = cp.nonzero(new_group)[0]  # shape = (num_groups,)

    # Assign subarray id (1..#subarray) to each element
    group_id = cp.cumsum(new_group)

    # For each element, the start index of its group
    start_pos = group_starts[group_id - 1]

    # local_index = global_index - start_pos
    return cp.arange(parents_length) - start_pos


def starts_to_offsets(starts, parents_length):
    offsets_dtype = starts.dtype

    if parents_length == 0:
        return cp.array([0], dtype=offsets_dtype)

    offsets = cp.empty(len(starts) + 1, dtype=offsets_dtype)
    offsets[:-1] = starts
    offsets[-1] = parents_length
    return offsets


def rearrange_by_parents(input_data, parents):
    order = cp.argsort(parents, kind="stable")
    return input_data[order]


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


def awkward_reduce_argmax_complex(
    result,
    input_data,
    offsets_data,
    starts,
    outlength,
):
    complex_dtype = infer_complex_dtype(input_data.dtype)

    input_complex = input_data.view(complex_dtype)

    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_argmax(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_complex[start_idx:end_idx]
        if start_idx == end_idx:
            return index_dtype(-1)
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


def awkward_reduce_argmin_complex(
    result,
    input_data,
    offsets_data,
    starts,
    outlength,
):
    index_dtype = normalize_index_dtype(offsets_data.dtype)

    complex_dtype = infer_complex_dtype(input_data.dtype)
    input_complex = input_data.view(complex_dtype)

    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_argmin(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]

        segment = input_complex[start_idx:end_idx]

        if len(segment) == 0:
            return index_dtype(-1)

        # np.argmin over complex uses lexicographic ordering (real, then imag)
        return np.argmin(segment) + start_idx

    segment_ids = CountingIterator(index_dtype(0))

    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_argmin,
        num_items=outlength,
    )


def awkward_axis_none_reduce_max(array):
    data_dtype = array.dtype

    # Initialize identity depending on dtype
    if data_dtype.kind in "iu":  # int / uint
        identity = cp.iinfo(data_dtype).min
    elif data_dtype.kind == "f":  # float
        identity = cp.finfo(data_dtype).min
    else:
        raise TypeError("Unsupported dtype for max reduction")

    def reduce_op(a, b):
        return a if a > b else b

    result_scalar = cp.empty(1, dtype=data_dtype)
    h_init = np.array([identity], dtype=data_dtype)

    reduce_into(
        d_in=array,
        d_out=result_scalar,
        op=reduce_op,
        num_items=len(array),
        h_init=h_init,
    )

    return result_scalar


def awkward_reduce_sum(
    result,
    input_data,
    offsets_data,
    outlength,
):
    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_sum(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        if start_idx == end_idx:
            return 0
        return np.sum(segment)

    segment_ids = CountingIterator(index_dtype(0))

    # TODO: replace with segmented_reduce once available/fixed in CCCL
    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_sum,
        num_items=outlength,
    )


def awkward_reduce_sum_bool(
    result,
    input_data,
    offsets_data,
    outlength,
):
    index_dtype = normalize_index_dtype(offsets_data.dtype)
    # Temporary workaround: bool reductions are unreliable on current backend
    if input_data.dtype == cp.bool_:
        input_data = input_data.view(cp.int8)

    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_sum(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        if start_idx == end_idx:
            return 0
        return np.any(segment)

    segment_ids = CountingIterator(index_dtype(0))

    # TODO: replace with segmented_reduce once available/fixed in CCCL
    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_sum,
        num_items=outlength,
    )


def awkward_reduce_sum_bool_complex(
    result,
    input_data,
    offsets_data,
    outlength,
):
    index_dtype = normalize_index_dtype(offsets_data.dtype)

    complex_dtype = infer_complex_dtype(input_data.dtype)
    input_complex = input_data.view(complex_dtype)

    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_sum(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]

        if start_idx == end_idx:
            return 0

        segment = input_complex[start_idx:end_idx]

        # any non-zero complex value -> True (1)
        return np.any(segment != complex_dtype(0))

    segment_ids = CountingIterator(index_dtype(0))

    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_sum,
        num_items=outlength,
    )


def awkward_reduce_sum_int32_bool_64(
    result,
    input_data,
    offsets_data,
    outlength,
):
    # Temporary workaround: bool instability in backend
    if input_data.dtype == cp.bool_:
        input_data = input_data.view(cp.int8)

    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_sum(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        if start_idx == end_idx:
            return 0
        return np.sum(segment)

    segment_ids = CountingIterator(index_dtype(0))

    # TODO: replace with segmented_reduce once available/fixed in CCCL
    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_sum,
        num_items=outlength,
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
    index_dtype = normalize_index_dtype(offsets_data.dtype)

    def segment_reduce_sum(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_complex[start_idx:end_idx]
        if start_idx == end_idx:
            return 0
        return np.sum(segment)

    segment_ids = CountingIterator(index_dtype(0))

    # TODO: replace with segmented_reduce once available/fixed in CCCL
    unary_transform(
        d_in=segment_ids,
        d_out=result_complex,
        op=segment_reduce_sum,
        num_items=outlength,
    )


def awkward_reduce_prod(
    result,
    input_data,
    offsets_data,
    outlength,
):
    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_prod(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        if start_idx == end_idx:
            return 1
        return np.prod(segment)

    segment_ids = CountingIterator(index_dtype(0))

    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_prod,
        num_items=outlength,
    )


def awkward_reduce_prod_complex(
    result,
    input_data,
    offsets_data,
    outlength,
):
    # Complex values arrive as a flat float32/float64 array of length 2*N
    # (real/imag interleaved). Caller in _reducers.py views complex128 -> float64
    # (or complex64 -> float32) and doubles the length. Re-view the buffers
    # back to complex dtype for the reduction.

    complex_dtype = infer_complex_dtype(input_data.dtype)

    input_complex = input_data.view(complex_dtype)
    result_complex = result.view(complex_dtype)

    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_prod(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_complex[start_idx:end_idx]
        if start_idx == end_idx:
            return complex_dtype(1.0 + 0.0j)
        return np.prod(segment)

    segment_ids = CountingIterator(index_dtype(0))

    # TODO: replace with segmented_reduce once available/fixed in CCCL
    unary_transform(
        d_in=segment_ids,
        d_out=result_complex,
        op=segment_reduce_prod,
        num_items=outlength,
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

    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_prod(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        return np.all(segment)

    segment_ids = CountingIterator(index_dtype(0))

    # TODO: replace with segmented_reduce once available/fixed in CCCL
    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_prod,
        num_items=outlength,
    )


def awkward_reduce_prod_bool_complex(
    result,
    input_data,
    offsets_data,
    outlength,
):
    # Complex values arrive as a flat float32/float64 array of length 2*N
    # (real/imag interleaved). Re-view the buffers back to complex dtype.

    complex_dtype = infer_complex_dtype(input_data.dtype)

    input_complex = input_data.view(complex_dtype)

    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_prod(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]

        # Identity for logical AND semantics
        if start_idx == end_idx:
            return True

        acc = True

        for i in range(start_idx, end_idx):
            acc = acc and (input_complex[i] != complex_dtype(0))

            # Early exit
            if not acc:
                break

        return acc

    segment_ids = CountingIterator(index_dtype(0))

    # TODO: replace with segmented_reduce once available/fixed in CCCL
    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_prod,
        num_items=outlength,
    )


def awkward_reduce_max(
    result,
    input_data,
    offsets_data,
    outlength,
    # initial value for the reduction
    identity,
):
    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_max(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        # Empty segment -> identity
        if start_idx == end_idx:
            return identity
        max_value = max(segment)
        # return identity if it is > than max_value from input_data
        return max(max_value, identity)

    segment_ids = CountingIterator(index_dtype(0))

    # TODO: replace with segmented_reduce once available/fixed in CCCL
    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_max,
        num_items=outlength,
    )


def awkward_reduce_max_complex(
    result,
    input_data,
    offsets_data,
    outlength,
    # initial value for the reduction (real component; imag identity is 0)
    identity,
):
    # Complex values arrive as a flat float32/float64 array of length 2*N
    # (real/imag interleaved). Caller in _reducers.py views complex128 -> float64
    # (or complex64 -> float32) and doubles the length. Re-view the buffers
    # back to complex dtype for the reduction.

    complex_dtype = infer_complex_dtype(input_data.dtype)

    input_complex = input_data.view(complex_dtype)
    result_complex = result.view(complex_dtype)

    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_max(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]

        # Identity semantics:
        # empty segment => (identity, 0)
        max_real = identity
        max_imag = 0.0

        for i in range(start_idx, end_idx):
            c = input_complex[i]

            real = c.real
            imag = c.imag

            # Lexicographic comparison on (real, imag)
            if real > max_real or (real == max_real and imag > max_imag):
                max_real = real
                max_imag = imag

        return complex_dtype(max_real + 1j * max_imag)

    segment_ids = CountingIterator(index_dtype(0))

    # TODO: replace with segmented_reduce once available/fixed in CCCL
    unary_transform(
        d_in=segment_ids,
        d_out=result_complex,
        op=segment_reduce_max,
        num_items=outlength,
    )


def awkward_reduce_min(
    result,
    input_data,
    offsets_data,
    outlength,
    # initial value for the reduction
    identity,
):
    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_min(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]

        # Empty segment -> identity
        if start_idx == end_idx:
            return identity
        min_value = min(segment)
        # return identity if it is < than min_value from input_data
        return min(min_value, identity)

    segment_ids = CountingIterator(index_dtype(0))

    # TODO: replace with segmented_reduce once available/fixed in CCCL
    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_min,
        num_items=outlength,
    )


def awkward_reduce_min_complex(
    result,
    input_data,
    offsets_data,
    outlength,
    # initial value for the reduction (real component; imag identity is 0)
    identity,
):
    # Complex values arrive as a flat float32/float64 array of length 2*N
    # (real/imag interleaved). Caller in _reducers.py views complex128 -> float64
    # (or complex64 -> float32) and doubles the length. Re-view the buffers
    # back to complex dtype for the reduction.

    complex_dtype = infer_complex_dtype(input_data.dtype)

    input_complex = input_data.view(complex_dtype)
    result_complex = result.view(complex_dtype)

    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_min(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]

        # Identity semantics:
        # empty segment => (identity, 0)
        min_real = identity
        min_imag = 0.0

        for i in range(start_idx, end_idx):
            c = input_complex[i]

            real = c.real
            imag = c.imag

            # Lexicographic comparison on (real, imag)
            if real < min_real or (real == min_real and imag < min_imag):
                min_real = real
                min_imag = imag

        return complex_dtype(min_real + 1j * min_imag)

    segment_ids = CountingIterator(index_dtype(0))

    # TODO: replace with segmented_reduce once available/fixed in CCCL
    unary_transform(
        d_in=segment_ids,
        d_out=result_complex,
        op=segment_reduce_min,
        num_items=outlength,
    )


def awkward_reduce_count_64(
    result,
    offsets_data,
    outlength,
):
    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_count(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]

        return end_idx - start_idx

    segment_ids = CountingIterator(index_dtype(0))

    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_count,
        num_items=outlength,
    )


def awkward_reduce_countnonzero(
    result,
    input_data,
    offsets_data,
    outlength,
):
    # Temporary workaround for bool instability
    if input_data.dtype == cp.bool_:
        input_data = input_data.view(cp.int8)

    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_countnonzero(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]

        count = 0

        for i in range(start_idx, end_idx):
            if input_data[i] != 0:
                count += 1

        return count

    segment_ids = CountingIterator(index_dtype(0))

    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_countnonzero,
        num_items=outlength,
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

    def fill(counter):
        i = counter // indexlength  # repetition id
        j = counter % indexlength  # position within repetition

        base = index[j]

        # shift only valid indices
        if base >= 0:
            return index_dtype(base + i * regularsize)
        else:
            return index_dtype(-1)

    counters = CountingIterator(index_dtype(0))

    unary_transform(
        d_in=counters,
        d_out=outindex,
        op=fill,
        num_items=output_size,
    )
