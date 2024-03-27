// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (index_in, offsets_in, mask_out, starts_out, stops_out, length, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(length + 1, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_Content_getitem_next_missing_jagged_getmaskstartstop_a", index_in.dtype, offsets_in.dtype, mask_out.dtype, starts_out.dtype, stops_out.dtype]))(grid, block, (index_in, offsets_in, mask_out, starts_out, stops_out, length, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_Content_getitem_next_missing_jagged_getmaskstartstop_b", index_in.dtype, offsets_in.dtype, mask_out.dtype, starts_out.dtype, stops_out.dtype]))(grid, block, (index_in, offsets_in, mask_out, starts_out, stops_out, length, scan_in_array, invocation_index, err_code))
// out["awkward_Content_getitem_next_missing_jagged_getmaskstartstop_a", {dtype_specializations}] = None
// out["awkward_Content_getitem_next_missing_jagged_getmaskstartstop_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_Content_getitem_next_missing_jagged_getmaskstartstop_a(
    T* index_in,
    C* offsets_in,
    U* mask_out,
    V* starts_out,
    W* stops_out,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      if (index_in[thread_id] >= 0) {
        scan_in_array[thread_id + 1] = 1;
      }
    }
  }
}

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_Content_getitem_next_missing_jagged_getmaskstartstop_b(
    T* index_in,
    C* offsets_in,
    U* mask_out,
    V* starts_out,
    W* stops_out,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      starts_out[thread_id] = offsets_in[scan_in_array[thread_id]];
      if (index_in[thread_id] < 0) {
        mask_out[thread_id] = -1;
        stops_out[thread_id] = offsets_in[scan_in_array[thread_id + 1]];
      } else {
        mask_out[thread_id] = thread_id;
        stops_out[thread_id] = offsets_in[scan_in_array[thread_id + 1]];
      }
    }
  }
}
