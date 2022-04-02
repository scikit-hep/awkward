// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tocarry, mask, length, validwhen, invocation_index, err_code) = args
//     scan_in_array = cupy.empty(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_ByteMaskedArray_getitem_nextcarry_a', tocarry.dtype, mask.dtype]))(grid, block, (tocarry, mask, validwhen, length, scan_in_array, invocation_index, err_code))
//     scan_in_array = inclusive_scan(grid, block, (scan_in_array, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_ByteMaskedArray_getitem_nextcarry_b', tocarry.dtype, mask.dtype]))(grid, block, (tocarry, mask, validwhen, length, scan_in_array, invocation_index, err_code))
// out["awkward_ByteMaskedArray_getitem_nextcarry_a", {dtype_specializations}] = None
// out["awkward_ByteMaskedArray_getitem_nextcarry_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C>
__global__ void
awkward_ByteMaskedArray_getitem_nextcarry_a(T* tocarry,
                                            const C* mask,
                                            bool validwhen,
                                            int64_t length,
                                            int64_t* scan_in_array,
                                            uint64_t invocation_index,
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

template <typename T, typename C>
__global__ void
awkward_ByteMaskedArray_getitem_nextcarry_b(T* tocarry,
                                            const C* mask,
                                            bool validwhen,
                                            int64_t length,
                                            int64_t* scan_in_array,
                                            uint64_t invocation_index,
                                            uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if ((mask[thread_id] != 0) == validwhen) {
        tocarry[scan_in_array[thread_id] - 1] = thread_id;
      }
    }
  }
}
