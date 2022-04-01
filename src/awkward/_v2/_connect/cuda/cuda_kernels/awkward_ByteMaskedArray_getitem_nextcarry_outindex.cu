// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, tocarry, outindex, mask, length, validwhen, invocation_index, err_code):
//     scan_in_array = cupy.empty(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_ByteMaskedArray_getitem_nextcarry_outindex_a']))(grid, block, (mask, scan_in_array, validwhen, length, invocation_index, err_code))
//     scan_in_array = inclusive_scan(grid, block, (scan_in_array, length, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_ByteMaskedArray_getitem_nextcarry_outindex_b']))(grid, block, (scan_in_array, tocarry, outindex, mask, validwhen, length, invocation_index, err_code))
// END PYTHON

__global__ void
awkward_ByteMaskedArray_getitem_nextcarry_outindex_a(int8_t* mask,
                                                     int64_t* scan_in_array,
                                                     bool validwhen,
                                                     int64_t length,
                                                     uint64_t* invocation_index,
                                                     uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if ((mask[thread_id] != 0) == validwhen) {
        scan_in_array[thread_id] = 1;
      } else {
        scan_in_array[thread_id] = 0;
      }
    }
  }
}

__global__ void
awkward_ByteMaskedArray_getitem_nextcarry_outindex_b(int64_t* scan_in_array,
                                                     int64_t* to_carry,
                                                     int64_t* outindex,
                                                     const int8_t* mask,
                                                     bool validwhen,
                                                     int64_t length,
                                                     uint64_t* invocation_index,
                                                     uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if ((mask[thread_id] != 0) != validwhen) {
        to_carry[scan_in_array[thread_id] - 1] = thread_id;
        outindex[thread_id] = scan_in_array[thread_id] - 1;
      } else {
        outindex[thread_id] = -1;
      }
    }
  }
}
