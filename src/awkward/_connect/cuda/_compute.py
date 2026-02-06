# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

# TODO: delete these after modifying argmin
from cuda.compute import (
    CountingIterator,
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


# the inputs for this function we get from file ~/awkward/src/awkward/_reducers.py:239, in ArgMax.apply(self, array, parents, starts, shifts, outlength)
def awkward_reduce_argmax(
    result,
    input_data,
    parents_data,
    parents_length,
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
    offsets = parents_to_offsets(parents_data, parents_length)
    start_o = offsets[:-1]
    end_o = offsets[1:]

    # Perform the segmented reduce
    # type_wrapper: cp.int64
    type_wrapper = cp.dtype(index_dtype).type
    segment_ids = CountingIterator(type_wrapper(0))
    # TODO: try using segmented_reduce instead when https://github.com/NVIDIA/cccl/issues/6171 is fixed
    unary_transform(segment_ids, result, segment_reduce_argmax, outlength)


# this function is called from ~/awkward/src/awkward/_reducers.py:161 (ArgMin.apply())
def awkward_reduce_argmin(
    result,
    input_data,
    parents_data,
    parents_length,
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
    offsets = parents_to_offsets(parents_data, parents_length)
    start_o = offsets[:-1]
    end_o = offsets[1:]

    # Perform the segmented reduce
    # type_wrapper: cp.int64
    type_wrapper = cp.dtype(index_dtype).type
    segment_ids = CountingIterator(type_wrapper(0))
    # TODO: try using segmented_reduce instead when https://github.com/NVIDIA/cccl/issues/6171 is fixed
    unary_transform(segment_ids, result, segment_reduce_argmin, outlength)
