# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

# TODO: delete these after modifying argmin
from cuda.compute import (
    CountingIterator,
    gpu_struct,
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
        fromptr,  # d_in_keys
        toptr,  # d_out_keys
        None,  # d_in_values (not sorting values, just keys)
        None,  # d_out_values
        num_items,  # num_items
        num_segments,  # num_segments
        start_offsets,  # start_offsets_in
        end_offsets,  # end_offsets_in
        order,  # order (ASCENDING or DESCENDING)
        None,  # stream (use default stream)
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
    unary_transform(segment_ids, result, segment_reduce_argmax, outlength)


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
    unary_transform(segment_ids, result, segment_reduce_argmin, outlength)


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
    unary_transform(segment_ids, result, segment_reduce_sum, outlength)


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
    unary_transform(segment_ids, result, segment_reduce_sum, outlength)


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
    unary_transform(segment_ids, result, segment_reduce_sum, outlength)


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
    unary_transform(segment_ids, result, segment_reduce_prod, outlength)


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
    unary_transform(segment_ids, result, segment_reduce_prod, outlength)


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
    unary_transform(segment_ids, result, segment_reduce_max, outlength)


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
    unary_transform(segment_ids, result, segment_reduce_max, outlength)

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
    unary_transform(segment_ids, result, segment_reduce_min, outlength)


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
    unary_transform(segment_ids, result, segment_reduce_count, outlength)


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


def awkward_localindex(toindex, length):
    # Fills toindex with [0, 1, 2, ..., length-1]
    def fill_local_index(i):
        return i

    segment_ids = CountingIterator(toindex.dtype.type(0))
    unary_transform(segment_ids, toindex, fill_local_index, length)
