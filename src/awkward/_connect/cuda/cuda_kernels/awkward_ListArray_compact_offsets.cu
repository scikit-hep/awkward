// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tooffsets, fromstarts, fromstops, length, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_compact_offsets_a", tooffsets.dtype, fromstarts.dtype, fromstops.dtype]))(grid, block, (tooffsets, fromstarts, fromstops, length, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_compact_offsets_b", tooffsets.dtype, fromstarts.dtype, fromstops.dtype]))(grid, block, (tooffsets, fromstarts, fromstops, length, scan_in_array, invocation_index, err_code))
// out["awkward_ListArray_compact_offsets_a", {dtype_specializations}] = None
// out["awkward_ListArray_compact_offsets_b", {dtype_specializations}] = None
// END PYTHON

enum class LISTARRAY_COMPACT_OFFSETS_ERRORS {
  ERROR_START_STOP,  // message: "stops[i] < starts[i]"
};

template <typename T, typename C, typename U>
__global__ void
awkward_ListArray_compact_offsets_a(
    T* tooffsets,
    const C* fromstarts,
    const U* fromstops,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      C start = fromstarts[thread_id];
      U stop = fromstops[thread_id];
      if (stop < start) {
        RAISE_ERROR(LISTARRAY_COMPACT_OFFSETS_ERRORS::ERROR_START_STOP)
      }
      scan_in_array[thread_id] = (stop - start);
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_ListArray_compact_offsets_b(
    T* tooffsets,
    const C* fromstarts,
    const U* fromstops,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    tooffsets[0] = 0;
    if (thread_id < length) {
      C start = fromstarts[thread_id];
      U stop = fromstops[thread_id];
      if (stop < start) {
        RAISE_ERROR(LISTARRAY_COMPACT_OFFSETS_ERRORS::ERROR_START_STOP)
      }
      tooffsets[thread_id + 1] = scan_in_array[thread_id];
    }
  }
}
