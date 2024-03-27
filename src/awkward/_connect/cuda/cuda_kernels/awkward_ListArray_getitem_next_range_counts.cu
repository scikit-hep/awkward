// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (total, fromoffsets, lenstarts, invocation_total, err_code) = args
//     scan_in_array = cupy.zeros(lenstarts, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_getitem_next_range_counts_a", total.dtype, fromoffsets.dtype]))(grid, block, (total, fromoffsets, lenstarts, scan_in_array, invocation_total, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_getitem_next_range_counts_b", total.dtype, fromoffsets.dtype]))(grid, block, (total, fromoffsets, lenstarts, scan_in_array, invocation_total, err_code))
// out["awkward_ListArray_getitem_next_range_counts_a", {dtype_specializations}] = None
// out["awkward_ListArray_getitem_next_range_counts_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C>
__global__ void
awkward_ListArray_getitem_next_range_counts_a(
    T* total,
    const C* fromoffsets,
    int64_t lenstarts,
    int64_t* scan_in_array,
    uint64_t invocation_total,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < lenstarts) {
      scan_in_array[thread_id] = (int64_t)fromoffsets[thread_id + 1] - fromoffsets[thread_id];
    }
  }
}

template <typename T, typename C>
__global__ void
awkward_ListArray_getitem_next_range_counts_b(
    T* total,
    const C* fromoffsets,
    int64_t lenstarts,
    int64_t* scan_in_array,
    uint64_t invocation_total,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    *total = lenstarts > 0 ? scan_in_array[lenstarts - 1] : 0;
  }
}
