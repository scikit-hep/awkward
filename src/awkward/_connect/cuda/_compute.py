# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from cuda.compute import (
    CountingIterator,
    DiscardIterator,
    TransformIterator,
    ZipIterator,
    inclusive_scan,
    reduce_into,
    unary_transform,
)
from cuda.compute.op import OpKind

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

    start_offsets = offsets[:-1]
    end_offsets = offsets[1:]

    order = SortOrder.ASCENDING if ascending else SortOrder.DESCENDING

    segmented_sort(
        d_in_keys=fromptr,
        d_out_keys=toptr,
        d_in_values=None,
        d_out_values=None,
        num_items=num_items,
        num_segments=num_segments,
        start_offsets_in=start_offsets,
        end_offsets_in=end_offsets,
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


# the inputs for this function we get from file ~/awkward/src/awkward/_reducers.py:239, in ArgMax.apply(self, array, parents, starts, shifts, outlength)
def awkward_reduce_argmax(
    result,
    input_data,
    parents_data,
    offsets_data,
    parents_length,
    starts,
    outlength,
):
    index_dtype = parents_data.dtype

    def segment_reduce_argmax(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        if len(segment) == 0:
            return -1
        # return a global index
        return np.argmax(segment) + start_idx

    # Prepare the start and end offsets
    offsets = starts_to_offsets(starts, parents_length)
    start_o = offsets[:-1]
    end_o = offsets[1:]

    # Perform the segmented reduce
    # type_wrapper is always cp.int64
    type_wrapper = cp.dtype(index_dtype).type
    segment_ids = CountingIterator(type_wrapper(0))
    # TODO: try using segmented_reduce instead when https://github.com/NVIDIA/cccl/issues/6171 is fixed
    unary_transform(
        d_in=segment_ids, d_out=result, op=segment_reduce_argmax, num_items=outlength
    )


# this function is called from ~/awkward/src/awkward/_reducers.py:161 (ArgMin.apply())
def awkward_reduce_argmin(
    result,
    input_data,
    parents_data,
    offsets_data,
    parents_length,
    starts,
    outlength,
):
    index_dtype = parents_data.dtype

    def segment_reduce_argmin(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        if len(segment) == 0:
            return -1
        # return a global index
        return np.argmin(segment) + start_idx

    # Prepare the start and end offsets
    offsets = starts_to_offsets(starts, parents_length)
    start_o = offsets[:-1]
    end_o = offsets[1:]

    # Perform the segmented reduce
    # type_wrapper is always cp.int64
    type_wrapper = cp.dtype(index_dtype).type
    segment_ids = CountingIterator(type_wrapper(0))
    # TODO: try using segmented_reduce instead when https://github.com/NVIDIA/cccl/issues/6171 is fixed
    unary_transform(
        d_in=segment_ids, d_out=result, op=segment_reduce_argmin, num_items=outlength
    )


def awkward_axis_none_reduce_max(array):
    data_dtype = array.dtype
    index_dtype = np.int64
    # initialize the minimum value depending on the dtype
    if data_dtype.kind in "iu":  # int/uint
        min = cp.iinfo(data_dtype).min
    elif data_dtype.kind == "f":  # float
        min = cp.finfo(data_dtype).min
    else:
        raise TypeError("Unsupported dtype to get the minimal value")

    def reduce_op(a, b):
        return max(a, b)

    result_scalar = cp.empty(1, dtype=index_dtype)
    h_init = np.array([min], dtype=index_dtype)
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
    parents_data,
    offsets_data,
    parents_length,
    outlength,
):
    index_dtype = parents_data.dtype

    def segment_reduce_sum(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        if len(segment) == 0:
            return 0
        return np.sum(segment)

    # sort input in case a user wants to call `CudaComputeKernel awkward_reduce_max` directly and specify unordered parents
    input_data = rearrange_by_parents(input_data, parents_data)

    # Prepare the start and end offsets
    # TODO: This should at least be starts_to_offsets
    offsets = parents_to_offsets(parents_data, parents_length)
    start_o = offsets[:-1]
    end_o = offsets[1:]

    # Perform the segmented reduce
    # type_wrapper: cp.int64
    type_wrapper = cp.dtype(index_dtype).type
    segment_ids = CountingIterator(type_wrapper(0))
    # TODO: try using segmented_reduce instead when https://github.com/NVIDIA/cccl/issues/6171 is fixed
    unary_transform(
        d_in=segment_ids, d_out=result, op=segment_reduce_sum, num_items=outlength
    )


def awkward_reduce_sum_complex(
    result,
    input_data,
    parents_data,
    offsets_data,
    parents_length,
    outlength,
):
    # Complex values arrive as a flat float32/float64 array of length 2*N
    # (real/imag interleaved). Caller in _reducers.py views complex128 -> float64
    # (or complex64 -> float32) and doubles the length. We re-view those buffers
    # back to the matching complex dtype so we can reuse the same segmented-sum
    # pattern as `awkward_reduce_sum`.
    if input_data.dtype == cp.float32:
        complex_dtype = cp.complex64
    else:
        complex_dtype = cp.complex128

    input_complex = input_data.view(complex_dtype)
    result_complex = result.view(complex_dtype)

    index_dtype = parents_data.dtype

    def segment_reduce_sum(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_complex[start_idx:end_idx]
        if len(segment) == 0:
            return complex_dtype(0)
        return np.sum(segment)

    # sort input in case a user wants to call `CudaComputeKernel awkward_reduce_sum_complex`
    # directly and specify unordered parents
    input_complex = rearrange_by_parents(input_complex, parents_data)

    # Prepare the start and end offsets
    # TODO: This should at least be starts_to_offsets
    offsets = parents_to_offsets(parents_data, parents_length)
    start_o = offsets[:-1]
    end_o = offsets[1:]

    # Perform the segmented reduce
    # type_wrapper: cp.int64
    type_wrapper = cp.dtype(index_dtype).type
    segment_ids = CountingIterator(type_wrapper(0))
    # TODO: try using segmented_reduce instead when https://github.com/NVIDIA/cccl/issues/6171 is fixed
    unary_transform(
        d_in=segment_ids,
        d_out=result_complex,
        op=segment_reduce_sum,
        num_items=outlength,
    )


# original implementation - currently bools don't work because of a bug on numba side
def awkward_reduce_sum_bool(
    result,
    input_data,
    parents_data,
    offsets_data,
    parents_length,
    outlength,
):
    # temporary workaround - fix this (currently bools don't work because of a bug on numba side)
    if input_data.dtype == cp.bool_:
        input_data = input_data.view(cp.int8)  # cast bool -> int8

    index_dtype = parents_data.dtype

    def segment_reduce_sum(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        return np.any(segment)

    # sort input in case a user wants to call `CudaComputeKernel awkward_reduce_max` directly and specify unordered parents
    input_data = rearrange_by_parents(input_data, parents_data)

    # Prepare the start and end offsets
    # TODO: This should at least be starts_to_offsets
    offsets = parents_to_offsets(parents_data, parents_length)
    start_o = offsets[:-1]
    end_o = offsets[1:]

    # Perform the segmented reduce
    # type_wrapper: cp.int64
    type_wrapper = cp.dtype(index_dtype).type
    segment_ids = CountingIterator(type_wrapper(0))
    # TODO: try using segmented_reduce instead when https://github.com/NVIDIA/cccl/issues/6171 is fixed
    unary_transform(
        d_in=segment_ids, d_out=result, op=segment_reduce_sum, num_items=outlength
    )


# this is the same as awkward_reduce_sum (we can possibly use it after the bug on numba side is fixed)
def awkward_reduce_sum_int32_bool_64(
    result,
    input_data,
    parents_data,
    offsets_data,
    parents_length,
    outlength,
):
    # temporary workaround - fix this (currently bools don't work because of a bug on numba side)
    if input_data.dtype == cp.bool_:
        input_data = input_data.view(cp.int8)  # cast bool -> int8

    index_dtype = parents_data.dtype

    def segment_reduce_sum(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        if len(segment) == 0:
            return 0
        return np.sum(segment)

    # sort input in case a user wants to call `CudaComputeKernel awkward_reduce_max` directly and specify unordered parents
    input_data = rearrange_by_parents(input_data, parents_data)

    # Prepare the start and end offsets
    # TODO: This should at least be starts_to_offsets
    offsets = parents_to_offsets(parents_data, parents_length)
    start_o = offsets[:-1]
    end_o = offsets[1:]

    # Perform the segmented reduce
    # type_wrapper: cp.int64
    type_wrapper = cp.dtype(index_dtype).type
    segment_ids = CountingIterator(type_wrapper(0))
    # TODO: try using segmented_reduce instead when https://github.com/NVIDIA/cccl/issues/6171 is fixed
    unary_transform(
        d_in=segment_ids, d_out=result, op=segment_reduce_sum, num_items=outlength
    )


def awkward_reduce_prod(
    result,
    input_data,
    parents_data,
    offsets_data,
    parents_length,
    outlength,
):
    index_dtype = parents_data.dtype

    def segment_reduce_prod(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        if len(segment) == 0:
            # that's what a cpu kernel passes for empty arrays (awkward-cpp/src/cpu-kernels/awkward_reduce_prod.cpp#L15)
            return 1
        return np.prod(segment)

    # sort input in case a user wants to call `CudaComputeKernel awkward_reduce_max` directly and specify unordered parents
    input_data = rearrange_by_parents(input_data, parents_data)

    # Prepare the start and end offsets
    # TODO: This should at least be starts_to_offsets
    offsets = parents_to_offsets(parents_data, parents_length)
    start_o = offsets[:-1]
    end_o = offsets[1:]

    # Perform the segmented reduce
    # type_wrapper: cp.int64
    type_wrapper = cp.dtype(index_dtype).type
    segment_ids = CountingIterator(type_wrapper(0))
    # TODO: try using segmented_reduce instead when https://github.com/NVIDIA/cccl/issues/6171 is fixed
    unary_transform(
        d_in=segment_ids, d_out=result, op=segment_reduce_prod, num_items=outlength
    )


def awkward_reduce_prod_bool(
    result,
    input_data,
    parents_data,
    offsets_data,
    parents_length,
    outlength,
):
    # temporary workaround - fix this (currently bools don't work because of a bug on numba side)
    if input_data.dtype == cp.bool_:
        input_data = input_data.view(cp.int8)  # cast bool -> int8

    index_dtype = parents_data.dtype

    def segment_reduce_prod(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        return np.all(segment)

    # sort input in case a user wants to call `CudaComputeKernel awkward_reduce_max` directly and specify unordered parents
    input_data = rearrange_by_parents(input_data, parents_data)

    # Prepare the start and end offsets
    # TODO: This should at least be starts_to_offsets
    offsets = parents_to_offsets(parents_data, parents_length)
    start_o = offsets[:-1]
    end_o = offsets[1:]

    # Perform the segmented reduce
    # type_wrapper: cp.int64
    type_wrapper = cp.dtype(index_dtype).type
    segment_ids = CountingIterator(type_wrapper(0))
    # TODO: try using segmented_reduce instead when https://github.com/NVIDIA/cccl/issues/6171 is fixed
    unary_transform(
        d_in=segment_ids, d_out=result, op=segment_reduce_prod, num_items=outlength
    )


def awkward_reduce_max(
    result,
    input_data,
    parents_data,
    offsets_data,
    parents_length,
    outlength,
    # the initial value for the reduction
    identity,
):
    index_dtype = parents_data.dtype

    def segment_reduce_max(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        if len(segment) == 0:
            return identity
        max_value = max(segment)
        # return identity if it is > than max_value from input_data
        return max(max_value, identity)

    # sort input in case a user wants to call `CudaComputeKernel awkward_reduce_max` directly and specify unordered parents
    # TODO: delete this? (it is only used in tests-cuda-kernels-explicit)
    input_data = rearrange_by_parents(input_data, parents_data)

    # Prepare the start and end offsets
    # TODO: This should at least be starts_to_offsets
    offsets = parents_to_offsets(parents_data, parents_length)
    start_o = offsets[:-1]
    end_o = offsets[1:]

    # Perform the segmented reduce
    # type_wrapper: cp.int64
    type_wrapper = cp.dtype(index_dtype).type
    segment_ids = CountingIterator(type_wrapper(0))
    # TODO: try using segmented_reduce instead when https://github.com/NVIDIA/cccl/issues/6171 is fixed
    unary_transform(
        d_in=segment_ids, d_out=result, op=segment_reduce_max, num_items=outlength
    )


def awkward_reduce_max_complex(
    result,
    input_data,
    parents_data,
    offsets_data,
    parents_length,
    outlength,
    # the initial value for the reduction (real component; imag identity is 0)
    identity,
):
    # Complex values arrive as a flat float32/float64 array of length 2*N
    # (real/imag interleaved). Caller in _reducers.py views complex128 -> float64
    # (or complex64 -> float32) and doubles the length. We re-view those buffers
    # back to the matching complex dtype so the segment reducer can read the
    # real/imag components via .real / .imag attribute access, avoiding the
    # gpu_struct pattern that failed in earlier attempts.
    if input_data.dtype == cp.float32:
        complex_dtype = cp.complex64
    else:
        complex_dtype = cp.complex128

    input_complex = input_data.view(complex_dtype)
    result_complex = result.view(complex_dtype)

    index_dtype = parents_data.dtype

    def segment_reduce_max(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        # Start with (identity, 0) per the CPU kernel; empty segments keep this.
        max_real = identity
        max_imag = 0.0
        for i in range(start_idx, end_idx):
            c = input_complex[i]
            x = c.real
            y = c.imag
            # Lex compare on (real, imag)
            if x > max_real or (x == max_real and y > max_imag):
                max_real = x
                max_imag = y
        return complex(max_real, max_imag)

    # sort input in case a user wants to call `CudaComputeKernel awkward_reduce_max_complex`
    # directly and specify unordered parents
    input_complex = rearrange_by_parents(input_complex, parents_data)

    # Prepare the start and end offsets
    # TODO: This should at least be starts_to_offsets
    offsets = parents_to_offsets(parents_data, parents_length)
    start_o = offsets[:-1]
    end_o = offsets[1:]

    # Perform the segmented reduce
    # type_wrapper: cp.int64
    type_wrapper = cp.dtype(index_dtype).type
    segment_ids = CountingIterator(type_wrapper(0))
    # TODO: try using segmented_reduce instead when https://github.com/NVIDIA/cccl/issues/6171 is fixed
    unary_transform(
        d_in=segment_ids,
        d_out=result_complex,
        op=segment_reduce_max,
        num_items=outlength,
    )


def awkward_reduce_min(
    result,
    input_data,
    parents_data,
    offsets_data,
    parents_length,
    outlength,
    # the initial value for the reduction
    identity,
):
    index_dtype = parents_data.dtype

    def segment_reduce_min(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        if len(segment) == 0:
            return identity
        min_value = min(segment)
        # return identity if it is < than min_value from input_data
        return min(min_value, identity)

    # sort input in case a user wants to call `CudaComputeKernel awkward_reduce_min` directly and specify unordered parents
    # TODO: delete this? (it is only used in tests-cuda-kernels-explicit)
    input_data = rearrange_by_parents(input_data, parents_data)

    # Prepare the start and end offsets
    # TODO: This should at least be starts_to_offsets
    offsets = parents_to_offsets(parents_data, parents_length)
    start_o = offsets[:-1]
    end_o = offsets[1:]

    # Perform the segmented reduce
    # type_wrapper: cp.int64
    type_wrapper = cp.dtype(index_dtype).type
    segment_ids = CountingIterator(type_wrapper(0))
    # TODO: try using segmented_reduce instead when https://github.com/NVIDIA/cccl/issues/6171 is fixed
    unary_transform(
        d_in=segment_ids, d_out=result, op=segment_reduce_min, num_items=outlength
    )


def awkward_reduce_count_64(
    result,
    parents_data,
    parents_length,
    outlength,
):
    index_dtype = parents_data.dtype

    def segment_reduce_count(segment_id):
        if segment_id > offsets_len:
            # (when we will pass offsets directly, this won't be needed)
            return 0
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        if end_idx < start_idx:
            # if there are empty arrays at the end (when we will pass offsets directly, this won't be needed)
            return 0
        return end_idx - start_idx

    # initialize all results values to be 0 by default
    result[:] = 0

    # Prepare the start and end offsets
    # TODO: This should at least be starts_to_offsets
    offsets = parents_to_offsets(parents_data, parents_length)
    start_o = offsets[:-1]
    end_o = offsets[1:]
    offsets_len = len(offsets) - 2

    # Perform the segmented reduce
    # type_wrapper: cp.int64
    type_wrapper = cp.dtype(index_dtype).type
    segment_ids = CountingIterator(type_wrapper(0))
    # TODO: try using segmented_reduce instead when https://github.com/NVIDIA/cccl/issues/6171 is fixed
    unary_transform(
        d_in=segment_ids, d_out=result, op=segment_reduce_count, num_items=outlength
    )


def awkward_reduce_countnonzero(
    result,
    input_data,
    parents_data,
    parents_length,
    outlength,
):
    # temporary workaround - fix this (currently bools don't work because of a bug on numba side)
    if input_data.dtype == cp.bool_:
        input_data = input_data.view(cp.int8)  # cast bool -> int8
    index_dtype = parents_data.dtype

    def segment_reduce_count_nonzero(segment_id):
        if segment_id > offsets_len:
            # (when we will pass offsets directly, this won't be needed)
            return 0
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        count = 0
        for i in range(end_idx - start_idx):
            if segment[i] != 0:
                count += 1
        return count

    # Prepare the start and end offsets
    # TODO: This should at least be starts_to_offsets
    offsets = parents_to_offsets(parents_data, parents_length)
    start_o = offsets[:-1]
    end_o = offsets[1:]
    offsets_len = len(offsets) - 2

    # Perform the segmented reduce
    # type_wrapper: cp.int64
    type_wrapper = cp.dtype(index_dtype).type
    segment_ids = CountingIterator(type_wrapper(0))
    # TODO: try using segmented_reduce instead when https://github.com/NVIDIA/cccl/issues/6171 is fixed
    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_count_nonzero,
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


# Fills tostarts and tostops with evenly spaced offsets of size `target` for each of the `length` lists
def awkward_index_rpad_and_clip_axis1(tostarts, tostops, target, length):
    def fill(i):
        tostarts[i] = tostarts.dtype.type(i * target)
        return tostarts.dtype.type(i * target + target)

    segment_ids = CountingIterator(tostarts.dtype.type(0))
    unary_transform(d_in=segment_ids, d_out=tostops, op=fill, num_items=length)


def awkward_missing_repeat(outindex, index, indexlength, repetitions, regularsize):
    """
    Repeats an index array `repetitions` times, adjusting valid (non-negative) indices
    by an offset(regularsize) each repetition.
    Missing values (-1) are preserved as-is across all repetitions.
    """

    index_dtype = outindex.dtype.type

    def fill_missing_repeat(counter):
        i = counter // indexlength  # number of repetition we're in
        j = counter % indexlength  # position within the current repetition
        val_offset = i * regularsize
        base = index[j]
        adjustment = index_dtype(val_offset) if base >= 0 else index_dtype(0)
        return base + adjustment

    output_size = repetitions * indexlength
    counters = CountingIterator(index_dtype(0))
    unary_transform(
        d_in=counters, d_out=outindex, op=fill_missing_repeat, num_items=output_size
    )


# For each element in a regular array of `length` sublists of fixed `size`,
# writes its position within its sublist (0, 1, ..., size-1) into toindex.
# Example: size=3, length=2 → toindex = [0, 1, 2, 0, 1, 2]
def awkward_RegularArray_localindex(toindex, size, length):
    if length == 0 or size == 0:
        return

    dtype = toindex.dtype.type

    def fill(k):
        return dtype(k % size)

    unary_transform(
        d_in=CountingIterator(dtype(0)),
        d_out=toindex,
        op=fill,
        num_items=length * size,
    )


# Broadcasts each element of fromadvanced across nextsize consecutive slots in toadvanced.
# Example: fromadvanced=[3, 7], nextsize=3 → toadvanced=[3, 3, 3, 7, 7, 7]
def awkward_RegularArray_getitem_next_range_spreadadvanced(
    toadvanced, fromadvanced, length, nextsize
):
    if length == 0 or nextsize == 0:
        return

    dtype = toadvanced.dtype.type

    def fill(k):
        return fromadvanced[k // nextsize]

    unary_transform(
        d_in=CountingIterator(dtype(0)),
        d_out=toadvanced,
        op=fill,
        num_items=length * nextsize,
    )


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

    dtype = tocarry.dtype.type

    def fill(k):
        i = k // nextsize
        j = k % nextsize
        return dtype(i * size + regular_start + j * step)

    unary_transform(
        d_in=CountingIterator(dtype(0)),
        d_out=tocarry,
        op=fill,
        num_items=length * nextsize,
    )


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
    if length == 0:
        return

    dtype = tocarry.dtype.type

    def fill_carry(i):
        return dtype(i * size + fromarray[fromadvanced[i]])

    adtype = toadvanced.dtype.type

    def fill_advanced(i):
        return adtype(i)

    unary_transform(
        d_in=CountingIterator(dtype(0)),
        d_out=tocarry,
        op=fill_carry,
        num_items=length,
    )
    unary_transform(
        d_in=CountingIterator(adtype(0)),
        d_out=toadvanced,
        op=fill_advanced,
        num_items=length,
    )


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

    dtype = tocarry.dtype.type

    def fill_both(k):
        j = k % lenarray
        i = k // lenarray
        tocarry[k] = dtype(i * size + fromarray[j])
        toadvanced[k] = dtype(j)
        return dtype(0)

    unary_transform(
        d_in=CountingIterator(dtype(0)),
        d_out=DiscardIterator(),
        op=fill_both,
        num_items=length * lenarray,
    )


# Expands a carry index over a RegularArray of fixed-size sublists.
# For each carry element fromcarry[i] and intra-sublist position j,
# tocarry[i*size + j] = fromcarry[i]*size + j
# Example: fromcarry=[2, 0], size=3 → tocarry=[6, 7, 8, 0, 1, 2]
def awkward_RegularArray_getitem_carry(tocarry, fromcarry, lencarry, size):
    if lencarry == 0 or size == 0:
        return

    dtype = tocarry.dtype.type

    def fill(k):
        i = k // size
        j = k % size
        return dtype(fromcarry[i] * size + j)

    unary_transform(
        d_in=CountingIterator(dtype(0)),
        d_out=tocarry,
        op=fill,
        num_items=lencarry * size,
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
    j = cp.arange(N, dtype=cp.int64)
    in_range = j < li  # GPU broadcast
    safe_j = cp.where(in_range, j, cp.int64(0))  # clamp out-of-range positions to 0
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

    dtype = toptr.dtype.type

    def fill(q):
        k = q // target
        j = q % target
        start = fromoffsets[k]
        count = fromoffsets[k + 1] - start
        if j < count:
            return dtype(fromptr[start + j])
        return dtype(0)

    unary_transform(
        d_in=CountingIterator(dtype(0)),
        d_out=toptr,
        op=fill,
        num_items=num_sublists * target,
    )


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
    shorter = min(target, size)
    dtype = toindex.dtype.type

    def fill(q):
        i = q // target
        j = q % target
        if j < shorter:
            return dtype(i * size + j)
        return dtype(-1)

    unary_transform(
        d_in=CountingIterator(dtype(0)),
        d_out=toindex,
        op=fill,
        num_items=length * target,
    )


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
    dtype = tocarry.dtype.type

    def _index(q):
        return dtype(q * size + regular_at)

    unary_transform(
        d_in=CountingIterator(dtype(0)),
        d_out=tocarry[:length],
        op=_index,
        num_items=length,
    )


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
    counts = fromoffsets[1:offsetslength].astype(cp.int64) - fromoffsets[:n].astype(
        cp.int64
    )
    if cp.any(counts < 0):
        raise ValueError(
            "offsets must be monotonically increasing in compiled CUDA code (awkward_ListOffsetArray_toRegularArray)"
        )
    if not cp.all(counts == counts[0]):
        raise ValueError(
            "cannot convert to RegularArray because subarray lengths are not regular in compiled CUDA code (awkward_ListOffsetArray_toRegularArray)"
        )
    size[:1] = counts[:1].astype(size.dtype)  # GPU→GPU


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
        tolength.dtype
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
    padded = cp.maximum(counts, target).astype(starts.dtype)
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
    dtype = toindex.dtype.type

    def fill(q):
        i = q // target
        j = q % target
        rangeval = fromoffsets[i + 1] - fromoffsets[i]
        shorter = rangeval if rangeval < target else target
        if j < shorter:
            return dtype(fromoffsets[i] + j)
        return dtype(-1)

    unary_transform(
        d_in=CountingIterator(dtype(0)),
        d_out=toindex,
        op=fill,
        num_items=length * target,
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
    cumulative_nones = cp.empty(length_indexes + 1, dtype=cp.int64)
    cumulative_nones[0] = 0
    if length_indexes > 0:
        none_mask = (noneindexes[:length_indexes] < 0).astype(cumulative_nones.dtype)
        inclusive_scan(
            d_in=none_mask,
            d_out=cumulative_nones[1 : length_indexes + 1],
            op=OpKind.PLUS,
            init_value=None,
            num_items=length_indexes,
        )
    idx = fromoffsets[:length_offsets]
    tooffsets[:length_offsets] = (idx - cumulative_nones[idx]).astype(tooffsets.dtype)


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
        d_in=cp.maximum(counts, target).astype(row_starts.dtype),
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
    tomin[:1] = cp.sum(cp.maximum(counts, target)).astype(tomin.dtype)


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
        tomin[:1] = result.astype(tomin.dtype)


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
    total[:1] = fromoffsets[lenstarts].astype(total.dtype) - fromoffsets[0].astype(total.dtype)


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
