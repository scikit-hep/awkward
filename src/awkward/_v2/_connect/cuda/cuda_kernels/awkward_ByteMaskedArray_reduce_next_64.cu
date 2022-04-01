// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, nextcarry, nextparents, outindex, mask, parents, length, validwhen, invocation_index, err_code):
//     scan_in_array = cupy.empty(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_ByteMaskedArray_reduce_next_64_a']))(grid, block, (scan_in_array, mask, validwhen, length, invocation_index, err_code))
//     scan_in_array = inclusive_scan(grid, block, (scan_in_array, length, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_ByteMaskedArray_reduce_next_64_b']))(grid, block, (nextcarry, nextparents, outindex, mask, parents, validwhen, length, scan_in_array, invocation_index, err_code))
// END PYTHON

__global__ void
awkward_ByteMaskedArray_reduce_next_64_a(int64_t* scan_in_array,
                                         const int8_t* mask,
                                         bool validwhen,
                                         int64_t length,
                                         uint64_t* invocation_index,
                                         uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if ((mask[thread_id] != 0) == validwhen) {
        scan_in_array[thread_id] = 1;
      }
    }
  }
}

__global__ void
awkward_ByteMaskedArray_reduce_next_64_b(int64_t* nextcarry,
                                         int64_t* nextparents,
                                         int64_t* outindex,
                                         const int8_t* mask,
                                         const int64_t* parents,
                                         bool validwhen,
                                         int64_t length,
                                         int64_t* scan_in_array,
                                         uint64_t* invocation_index,
                                         uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if ((mask[thread_id] != 0) == validwhen) {
        nextcarry[scan_in_array[thread_id] - 1] = thread_id;
        nextparents[scan_in_array[thread_id] - 1] = parents[thread_id];
        outindex[thread_id] = scan_in_array[thread_id] - 1;
      } else {
        outindex[thread_id] = -1;
      }
    }
  }
}
