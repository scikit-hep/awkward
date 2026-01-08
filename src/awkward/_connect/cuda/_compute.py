# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import cupy as cp
from cuda.compute import ZipIterator, gpu_struct, segmented_reduce

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
    ak_array = gpu_struct({
        "data": input_data.dtype.type,
        "local_index": cp.int64,
    })

    # compare the values of the arrays
    def max_op(a: ak_array, b: ak_array):
        return a if a.data > b.data else b

    # use a helper function to get the local indices
    # local_indices = local_idx_from_parents(parents_data, parents_length)

    # use global indices instead?
    global_indices = cp.arange(0, parents_length + 1, dtype=cp.int64)

    # Combine data and their indices into a single structure
    input_struct = ZipIterator(input_data, global_indices)
    # alternative way
    # input_struct = cp.stack((input_data, global_indices), axis=1).view(ak_array.dtype)

    # Prepare the start and end offsets
    offsets = parents_to_offsets(parents_data, parents_length)
    start_o = offsets[:-1]
    end_o = offsets[1:]

    # Prepare the output array
    _result = result
    _result = cp.concatenate((result, result))
    _result = _result.view(ak_array.dtype)

    # alternative way
    # _result = cp.zeros([outlength], dtype= ak_array.dtype)

    # Initial value for the reduction
    # min value gets transformed to input_data.dtype automatically?
    min = cp.iinfo(cp.int64).min
    h_init = ak_array(min, min)

    # Perform the segmented reduce
    segmented_reduce(input_struct, _result, start_o, end_o, max_op, h_init, outlength)

    # TODO: here converts float to int too, fix this?
    _result = _result.view(cp.int64).reshape(-1, 2)
    _result = _result[:, 1]

    # pass the result outside the function
    result_v = result.view()
    result_v[...] = _result
