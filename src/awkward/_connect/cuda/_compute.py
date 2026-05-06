# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from cuda.compute import (
    CountingIterator,
    DiscardIterator,
    gpu_struct,
    inclusive_scan,
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


# original implementation of `awkward_reduce_max_complex` (doesn't work - keep for archive)
def awkward_reduce_max_complex(
    result,
    input_data,
    parents_data,
    offsets_data,
    parents_length,
    outlength,
    # the initial value for the reduction
    identity,
):
    # print("outlength", outlength)
    # print(input_data)
    # print(parents_data)
    index_dtype = parents_data.dtype
    data_dtype = input_data.dtype.type

    complex_array = gpu_struct(
        {
            "real_data": data_dtype,
            "imag_data": data_dtype,
        }
    )
    result = result.view(complex_array.dtype)

    def segment_reduce_max(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        # segment = input_data[start_idx:end_idx]
        if end_idx <= start_idx:
            return identity  # empty segment

        max_real = identity
        max_imag = type_wrapper(0)

        for i in range(start_idx, end_idx):
            x = input_data[i * 2]
            y = input_data[i * 2 + 1]
            if x > max_real or (x == max_real and y > max_imag):
                max_real = x
                max_imag = y

        # max_value = max(segment)
        # return identity if it is > than max_value from input_data
        # print(complex_array(max_real, max_imag))
        return complex_array(max_real, max_imag)

    # sort input in case a user wants to call `CudaComputeKernel awkward_reduce_max` directly and specify unordered parents
    # TODO: delete this? (it is only used in tests-cuda-kernels-explicit)
    input_data = rearrange_by_parents(input_data, parents_data)

    # Prepare the start and end offsets
    # TODO: This should at least be starts_to_offsets
    offsets = parents_to_offsets(parents_data, parents_length)
    # print(offsets)
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

    # print("this is the result:", result)


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
    unary_transform(segment_ids, result, segment_reduce_count_nonzero, outlength)


# Overlays a mask onto an index array: masked positions become -1, unmasked positions keep their original index value.
def awkward_IndexedArray_overlay_mask(toindex, mask, fromindex, length):
    def transform(i):
        return -1 if mask[i] else fromindex[i]

    indices = CountingIterator(cp.int64(0))
    unary_transform(indices, toindex, transform, length)


# Skips masked (-1) entries and packs the remaining valid entries into nextcarry and nextparents, tracking where each ended up in outindex.
def awkward_IndexedArray_reduce_next_64(
    nextcarry, nextparents, outindex, index, parents, length
):
    if length == 0:
        return

    # Compute cumulative count of valid (non-negative) indices to determine compact output positions
    # this needs to be done before going through all the indices in parallel later
    scan = cp.cumsum(index >= 0)

    def scatter_and_fill(i):
        if index[i] >= 0:
            # Map valid entry to its compacted position
            k = scan[i] - 1
            nextcarry[k] = index[i]
            nextparents[k] = parents[i]
            return k
        # Masked entries get -1 in outindex
        return -1

    indices = CountingIterator(cp.int64(0))
    unary_transform(indices, outindex, scatter_and_fill, length)


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
    unary_transform(indices, _, scatter, length)


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
        CountingIterator(cp.int64(0)), DiscardIterator(), scatter_and_update, length
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

    unary_transform(CountingIterator(cp.int64(0)), DiscardIterator(), fill, total_size)


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

    def transform(i):
        if fromtags[i] == fromwhich:
            totags[i] = towhich
            toindex[i] = fromindex[i] + base
        return 0  # discarded

    indices = CountingIterator(cp.int64(0))
    unary_transform(indices, DiscardIterator(), transform, length)


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
    def fill_list(i):
        start = starts[i]
        stop = stops[i]
        for j in range(start, stop):
            tocarry[fromoffsets[i] + j - start] = j
        return 0

    unary_transform(CountingIterator(cp.int64(0)), DiscardIterator(), fill_list, length)


# For each segment i, it fills toindex with the local position of each element within that segment — i.e. 0, 1, 2, ...
# Example:
# offsets = [0, 3, 5]
# toindex = [0, 1, 2, 0, 1]
def awkward_ListArray_localindex(toindex, offsets, length):
    if length == 0:
        return

    starts = offsets[:length]
    stops = offsets[1 : length + 1]

    def fill(i):
        start = starts[i]
        stop = stops[i]
        for j in range(start, stop):
            toindex[j] = j - start
        return 0

    unary_transform(CountingIterator(cp.int64(0)), DiscardIterator(), fill, length)


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
        sizes,
        tooffsets[1 : length + 1],
        lambda a, b: a + b,
        cp.array([0], dtype=tooffsets.dtype),
        length,
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
    unary_transform(CountingIterator(cp.int64(0)), counts, combinations_len, length)

    # Convert counts to offsets:
    # tooffsets[i+1] = sum(counts[0..i])
    inclusive_scan(
        counts,
        tooffsets[1 : length + 1],
        lambda a, b: a + b,
        cp.array([0], dtype=tooffsets.dtype),
        length,
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
    unary_transform(CountingIterator(cp.int64(0)), counts, combinations_len, length)

    offsets = cp.empty(length + 1, dtype=cp.int64)
    offsets[0] = 0
    inclusive_scan(
        counts,
        offsets[1:],
        lambda a, b: a + b,
        cp.array([0], dtype=cp.int64),
        length,
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
        carry_arrays.append(cp.ndarray(totallen, dtype=cp.int64, memptr=memptr))

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
            CountingIterator(cp.int64(0)),
            DiscardIterator(),
            make_pass(k, carry_arrays[k]),
            totallen,
        )

    toindex[:n] = totallen
