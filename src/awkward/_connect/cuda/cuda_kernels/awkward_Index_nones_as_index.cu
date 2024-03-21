// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toindex, length, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(length, dtype=cupy.int64)
//     scan_in_array_n_non_null = cupy.zeros(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_Index_nones_as_index_a", toindex.dtype]))(grid, block, (toindex, length, scan_in_array, scan_in_array_n_non_null, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     scan_in_array_n_non_null = cupy.cumsum(scan_in_array_n_non_null)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_Index_nones_as_index_b", toindex.dtype]))(grid, block, (toindex, length, scan_in_array, scan_in_array_n_non_null, invocation_index, err_code))
// out["awkward_Index_nones_as_index_a", {dtype_specializations}] = None
// out["awkward_Index_nones_as_index_b", {dtype_specializations}] = None
// END PYTHON

template <typename T>
__global__ void
awkward_Index_nones_as_index_a(
    T* toindex,
    int64_t length,
    int64_t* scan_in_array,
    int64_t* scan_in_array_n_non_null,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      if (toindex[thread_id] != -1) {
        scan_in_array[thread_id] = 1;
      }
      else {
        scan_in_array_n_non_null[thread_id] = 1;
      }
    }
  }
}

template <typename T>
__global__ void
awkward_Index_nones_as_index_b(
    T* toindex,
    int64_t length,
    int64_t* scan_in_array,
    int64_t* scan_in_array_n_non_null,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t n_non_null = scan_in_array[length - 1];
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      toindex[thread_id] == -1 ? toindex[thread_id] = (n_non_null + scan_in_array_n_non_null[thread_id] - 1): toindex[thread_id];
    }
  }
}
