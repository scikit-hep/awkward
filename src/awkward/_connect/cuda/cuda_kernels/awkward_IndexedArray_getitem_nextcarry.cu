// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class INDEXEDARRAY_GETITEM_NEXTCARRY_ERRORS {
  IND_OUT_OF_RANGE,  // message: "index out of range"
};

// BEGIN PYTHON
// def f(grid, block, args):
//     (tocarry, fromindex, lenindex, lencontent, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(lenindex, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_getitem_nextcarry_a", tocarry.dtype, fromindex.dtype]))(grid, block, (tocarry, fromindex, lenindex, lencontent, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_getitem_nextcarry_b", tocarry.dtype, fromindex.dtype]))(grid, block, (tocarry, fromindex, lenindex, lencontent, scan_in_array, invocation_index, err_code))
// out["awkward_IndexedArray_getitem_nextcarry_a", {dtype_specializations}] = None
// out["awkward_IndexedArray_getitem_nextcarry_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C>
__global__ void
awkward_IndexedArray_getitem_nextcarry_a(
    T* tocarry,
    const C* fromindex,
    int64_t lenindex,
    int64_t lencontent,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < lenindex) {
      C j = fromindex[thread_id];
      if (j < 0 || j >= lencontent) {
        RAISE_ERROR(INDEXEDARRAY_GETITEM_NEXTCARRY_ERRORS::IND_OUT_OF_RANGE)
      } else if (j >= 0) {
        scan_in_array[thread_id] = 1;
      }
    }
  }
}

template <typename T, typename C>
__global__ void
awkward_IndexedArray_getitem_nextcarry_b(
    T* tocarry,
    const C* fromindex,
    int64_t lenindex,
    int64_t lencontent,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < lenindex) {
      C j = fromindex[thread_id];
      if (j < 0 || j >= lencontent) {
        RAISE_ERROR(INDEXEDARRAY_GETITEM_NEXTCARRY_ERRORS::IND_OUT_OF_RANGE)
      } else if (j >= 0) {
        tocarry[scan_in_array[thread_id] - 1] = j;
      }
    }
  }
}
