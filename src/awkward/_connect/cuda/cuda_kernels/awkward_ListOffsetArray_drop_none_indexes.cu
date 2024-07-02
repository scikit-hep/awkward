// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tooffsets, noneindexes, fromoffsets, length_offsets, length_indexes, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(length_offsets, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_drop_none_indexes_a", tooffsets.dtype, noneindexes.dtype, fromoffsets.dtype]))(grid, block, (tooffsets, noneindexes, fromoffsets, length_offsets, length_indexes, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_drop_none_indexes_b", tooffsets.dtype, noneindexes.dtype, fromoffsets.dtype]))(grid, block, (tooffsets, noneindexes, fromoffsets, length_offsets, length_indexes, scan_in_array, invocation_index, err_code))
// out["awkward_ListOffsetArray_drop_none_indexes_a", {dtype_specializations}] = None
// out["awkward_ListOffsetArray_drop_none_indexes_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_ListOffsetArray_drop_none_indexes_a(
    T* tooffsets,
    const C* noneindexes,
    const U* fromoffsets,
    int64_t length_offsets,
    int64_t length_indexes,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    T nr_of_nones = 0;
    int64_t offset1 = 0;
    int64_t offset2 = 0;

    if (thread_id < length_offsets) {
      if (thread_id > 0) {
        offset1 = fromoffsets[thread_id - 1];
      }
      offset2 = fromoffsets[thread_id];
      for (int j = offset1 + threadIdx.y; j < offset2; j += blockDim.y) {
        if (noneindexes[j] < 0) {
          nr_of_nones++;
        }
      }
      scan_in_array[thread_id] = nr_of_nones;
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_ListOffsetArray_drop_none_indexes_b(
    T* tooffsets,
    const C* noneindexes,
    const U* fromoffsets,
    int64_t length_offsets,
    int64_t length_indexes,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length_offsets) {
      T nr_of_nones = scan_in_array[thread_id];
      tooffsets[thread_id] = fromoffsets[thread_id] - nr_of_nones;
    }
  }
}
