// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (index, starts_in, stops_in, starts_out, stops_out, length, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_MaskedArray_getitem_next_jagged_project_a", index.dtype, starts_in.dtype, stops_in.dtype, starts_out.dtype, stops_out.dtype]))(grid, block, (index, starts_in, stops_in, starts_out, stops_out, length, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_MaskedArray_getitem_next_jagged_project_b", index.dtype, starts_in.dtype, stops_in.dtype, starts_out.dtype, stops_out.dtype]))(grid, block, (index, starts_in, stops_in, starts_out, stops_out, length, scan_in_array, invocation_index, err_code))
// out["awkward_MaskedArray_getitem_next_jagged_project_a", {dtype_specializations}] = None
// out["awkward_MaskedArray_getitem_next_jagged_project_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_MaskedArray_getitem_next_jagged_project_a(
    T* index,
    C* starts_in,
    U* stops_in,
    V* starts_out,
    W* stops_out,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if (index[thread_id] >= 0) {
        scan_in_array[thread_id] = 1;
      }
    }
  }
}

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_MaskedArray_getitem_next_jagged_project_b(
    T* index,
    C* starts_in,
    U* stops_in,
    V* starts_out,
    W* stops_out,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if (index[thread_id] >= 0) {
        starts_out[scan_in_array[thread_id] - 1] = starts_in[thread_id];
        stops_out[scan_in_array[thread_id] - 1] = stops_in[thread_id];
      }
    }
  }
}
