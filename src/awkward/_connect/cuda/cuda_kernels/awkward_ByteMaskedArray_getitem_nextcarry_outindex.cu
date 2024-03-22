// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tocarry, outindex, mask, length, validwhen, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_ByteMaskedArray_getitem_nextcarry_outindex_a', tocarry.dtype, outindex.dtype, mask.dtype]))(grid, block, (tocarry, outindex, mask, length, validwhen, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_ByteMaskedArray_getitem_nextcarry_outindex_b', tocarry.dtype, outindex.dtype, mask.dtype]))(grid, block, (tocarry, outindex, mask, length, validwhen, scan_in_array, invocation_index, err_code))
// out["awkward_ByteMaskedArray_getitem_nextcarry_outindex_a", {dtype_specializations}] = None
// out["awkward_ByteMaskedArray_getitem_nextcarry_outindex_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_ByteMaskedArray_getitem_nextcarry_outindex_a(
    T* tocarry,
    C* outindex,
    const U* mask,
    int64_t length,
    bool validwhen,
    int64_t* scan_in_array,
    uint64_t invocation_index,
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

template <typename T, typename C, typename U>
__global__ void
awkward_ByteMaskedArray_getitem_nextcarry_outindex_b(
    T* tocarry,
    C* outindex,
    const U* mask,
    int64_t length,
    bool validwhen,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if ((mask[thread_id] != 0) == validwhen) {
        tocarry[scan_in_array[thread_id] - 1] = thread_id;
        outindex[thread_id] = (T)scan_in_array[thread_id] - 1;
      } else {
        outindex[thread_id] = -1;
      }
    }
  }
}
