// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toindex, fromstarts, fromstops, tostarts, tostops, target, length, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_rpad_axis1_a", toindex.dtype, fromstarts.dtype, fromstops.dtype, tostarts.dtype, tostops.dtype]))(grid, block, (toindex, fromstarts, fromstops, tostarts, tostops, target, length, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_rpad_axis1_b", toindex.dtype, fromstarts.dtype, fromstops.dtype, tostarts.dtype, tostops.dtype]))(grid, block, (toindex, fromstarts, fromstops, tostarts, tostops, target, length, scan_in_array, invocation_index, err_code))
// out["awkward_ListArray_rpad_axis1_a", {dtype_specializations}] = None
// out["awkward_ListArray_rpad_axis1_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_ListArray_rpad_axis1_a(
    T* toindex,
    const C* fromstarts,
    const U* fromstops,
    V* tostarts,
    W* tostops,
    int64_t target,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      int64_t rangeval = fromstops[thread_id] - fromstarts[thread_id];
      scan_in_array[thread_id] = (target > rangeval) ? target : rangeval;
    }
  }
}

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_ListArray_rpad_axis1_b(
    T* toindex,
    const C* fromstarts,
    const U* fromstops,
    V* tostarts,
    W* tostops,
    int64_t target,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      int64_t offset = 0;

      if (thread_id > 0) {
        offset = scan_in_array[thread_id - 1];
      }
      tostarts[thread_id] = offset;
      int64_t rangeval = fromstops[thread_id] - fromstarts[thread_id];
      for (int64_t j = 0; j < rangeval; j++) {
        toindex[offset + j] = fromstarts[thread_id] + j;
       }
       for (int64_t j = rangeval; j < target; j++) {
        toindex[offset + j] = -1;
       }
      tostops[thread_id] = scan_in_array[thread_id];
    }
  }
}
