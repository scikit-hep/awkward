// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(index_in, offsets_in, mask_out, starts_out, stops_out, length):
//     scan_in_array = cupy.empty(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function("awkward_Content_getitem_next_missing_jagged_getmaskstartstop_a")(grid, block, (index_in, scan_in_array, length, invocation_index, err_code))
//     scan_in_array = inclusive_scan(scan_in_array, length)
//     cuda_kernel_templates.get_function("awkward_Content_getitem_next_missing_jagged_getmaskstartstop_b")(grid, block, (scan_in_array, index_in, offsets_in, mask_out, starts_out, stops_out, length, invocation_index, err_code))
// END PYTHON

__global__ void
awkward_Content_getitem_next_missing_jagged_getmaskstartstop_a(
    int64_t* index_in,
    int64_t* scan_in_array,
    int64_t length,
    uint64_t* invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if (index_in[thread_id] >= 0) {
        scan_in_array[thread_id] = 1;
      } else {
        scan_in_array[thread_id] = 0;
      }
    }
  }
}

__global__ void
awkward_Content_getitem_next_missing_jagged_getmaskstartstop_b(
    int64_t* scan_in_array,
    int64_t* index_in,
    int64_t* offsets_in,
    int64_t* mask_out,
    int64_t* starts_out,
    int64_t* stops_out,
    int64_t length,
    uint64_t* invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      int64_t pre_in = scan_in_array[thread_id] - 1;
      starts_out[thread_id] = offsets_in[pre_in];

      if (index_in[thread_id] < 0) {
        mask_out[thread_id] = -1;
        stops_out[thread_id] = offsets_in[pre_in];
      } else {
        mask_out[thread_id] = thread_id;
        stops_out[thread_id] = offsets_in[pre_in + 1];
      }
    }
  }
}
