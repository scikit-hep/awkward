// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tomin, fromstarts, fromstops, lenstarts, invocation_index, err_code) = args
//     scan_in_array = cupy.empty(lenstarts, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_min_range_a", tomin.dtype, fromstarts.dtype, fromstops.dtype]))(grid, block, (tomin, fromstarts, fromstops, lenstarts, scan_in_array, invocation_index, err_code))
//     if lenstarts > 0:
//         scan_in_array[0] = cupy.min(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_min_range_b", tomin.dtype, fromstarts.dtype, fromstops.dtype]))(grid, block, (tomin, fromstarts, fromstops, lenstarts, scan_in_array, invocation_index, err_code))
// out["awkward_ListArray_min_range_a", {dtype_specializations}] = None
// out["awkward_ListArray_min_range_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_ListArray_min_range_a(
    T* tomin,
    const C* fromstarts,
    const U* fromstops,
    int64_t lenstarts,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < lenstarts) {
      if (thread_id == 0) {
        scan_in_array[thread_id] = fromstops[0] - fromstarts[0];
      }
      else {
        int64_t rangeval = fromstops[thread_id] - fromstarts[thread_id];
        scan_in_array[thread_id] = rangeval;
      }
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_ListArray_min_range_b(
    T* tomin,
    const C* fromstarts,
    const U* fromstops,
    int64_t lenstarts,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    if (lenstarts > 0) {
      *tomin = scan_in_array[0];
    }
  }
}
