// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, nextshifts, mask, length, valid_when, invocation_index, err_code):
//     scan_in_array_k = cupy.empty(length, dtype=cupy.int64)
//     scan_in_array_nullsum = cupy.empty(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function("awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64_a")(grid, block, (scan_in_array_k, scan_in_array_nullsum, mask, valid_when, length, invocation_index, err_code))
//     scan_in_array_k = inclusive_scan(scan_in_array_k, length)
//     scan_in_array_nullsum = inclusive_scan(scan_in_array_nullsum, length)
//     cuda_kernel_templates.get_function("awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64_b")(grid, block, (nextshifts, mask, valid_when, length, scan_in_array_k, scan_in_array_nullsum, invocation_index, err_code))
// END PYTHON

__global__ void
awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64_a(
    int64_t* scan_in_array_k,
    int64_t* scan_in_array_nullsum,
    const int8_t* mask,
    bool valid_when,
    int64_t length,
    uint64_t* invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if ((mask[thread_id] != 0) == (valid_when != 0)) {
        scan_in_array_k[thread_id] = 1;
        scan_in_array_nullsum[thread_id] = 0;
      } else {
        scan_in_array_nullsum[thread_id] = 1;
        scan_in_array_k[thread_id] = 0;
      }
    }
  }
}
__global__ void
awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64_b(
    int64_t* nextshifts,
    const int8_t* mask,
    bool valid_when,
    int64_t length,
    int64_t* scan_in_array_k,
    int64_t* scan_in_array_nullsum,
    uint64_t* invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if ((mask[thread_id] != 0) == (valid_when != 0)) {
        nextshifts[scan_in_array_k[thread_id] - 1] =
            scan_in_array_nullsum[thread_id] - 1;
      }
    }
  }
}
