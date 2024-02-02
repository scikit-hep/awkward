// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toarray, fromarray, lenarray, size, invocation_index, err_code) = args
//     scan_in_array = cupy.empty(lenarray, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_RegularArray_getitem_next_array_regularize_a', toarray.dtype, fromarray.dtype]))(grid, block, (toarray, fromarray, lenarray, size, scan_in_array, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_RegularArray_getitem_next_array_regularize_b', toarray.dtype, fromarray.dtype]))(grid, block, (toarray, fromarray, lenarray, size, scan_in_array, invocation_index, err_code))
// out["awkward_RegularArray_getitem_next_array_regularize_a", {dtype_specializations}] = None
// out["awkward_RegularArray_getitem_next_array_regularize_b", {dtype_specializations}] = None
// END PYTHON

enum class REGULARARRAY_GETITEM_NEXT_ARRAY_REGULARIZE_ERRORS {
  IND_OUT_OF_RANGE  // message: "index out of range"
};

template <typename T, typename C>
__global__ void
awkward_RegularArray_getitem_next_array_regularize_a(
    T* toarray,
    const C* fromarray,
    int64_t lenarray,
    int64_t size,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < lenarray) {
      scan_in_array[thread_id] = fromarray[thread_id];
      if (scan_in_array[thread_id] < 0) {
        scan_in_array[thread_id] = fromarray[thread_id] + size;
      }
      if (!(0 <= scan_in_array[thread_id]  &&  scan_in_array[thread_id] < size)) {
        RAISE_ERROR(REGULARARRAY_GETITEM_NEXT_ARRAY_REGULARIZE_ERRORS::IND_OUT_OF_RANGE)
      }
    }
  }
}

template <typename T, typename C>
__global__ void
awkward_RegularArray_getitem_next_array_regularize_b(
    T* toarray,
    const C* fromarray,
    int64_t lenarray,
    int64_t size,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < lenarray) {
      toarray[thread_id] = scan_in_array[thread_id];
      if (!(0 <= toarray[thread_id]  &&  toarray[thread_id] < size)) {
        RAISE_ERROR(REGULARARRAY_GETITEM_NEXT_ARRAY_REGULARIZE_ERRORS::IND_OUT_OF_RANGE)
      }
    }
  }
}
