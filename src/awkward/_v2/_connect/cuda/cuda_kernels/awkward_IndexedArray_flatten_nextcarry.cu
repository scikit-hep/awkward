// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class AWKWARD_INDEXEDARRAY_FLATTEN_NEXTCARRY_ERRORS {
  IND_OUT_OF_RANGE // message: "index out of range"
};

// BEGIN PYTHON
// def f(grid, block, tocarry, fromindex, lenindex, lencontent, invocation_index, err_code):
//     scan_in_array = cupy.empty(lenindex, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_flatten_nextcarry_a", fromindex.dtype]))(grid, block, (tocarry, fromindex, scan_in_array, lenindex, lencontent, invocation_index, err_code))
//     scan_in_array = inclusive_scan(scan_in_array, lenindex)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_flatten_nextcarry_b", fromindex.dtype, tocarry.dtype]))(grid, block, (tocarry, fromindex, scan_in_array, lenindex, lencontent, invocation_index, err_code))
// out["awkward_IndexedArray_flatten_nextcarry_a", {dtype_specializations}] = None
// out["awkward_IndexedArray_flatten_nextcarry_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C>
__global__ void
awkward_IndexedArray_flatten_nextcarry_a(
  T* tocarry,
  const C* fromindex,
  int64_t* scan_in_array,
  int64_t lenindex,
  int64_t lencontent,
  uint64_t* invocation_index,
  uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < lenindex) {
      C j = fromindex[thread_id];
      if (j >= lencontent) {
        RAISE_ERROR(AWKWARD_INDEXEDARRAY_FLATTEN_NEXTCARRY_ERRORS::IND_OUT_OF_RANGE)
      }
      else if (j >= 0) {
        scan_in_array[thread_id] = 1;
      }
      else {
        scan_in_array[thread_id] = 0;
      }
    }
  }
}

template <typename T, typename C>
__global__
void awkward_IndexedArray_flatten_nextcarry_b(
    T* tocarry,
    const C* fromindex,
    int64_t* scan_in_array,
    int64_t lenindex,
    int64_t lencontent,
    uint64_t* invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < lenindex) {
      C j = fromindex[thread_id];
      if (j >= lencontent) {
        RAISE_ERROR(AWKWARD_INDEXEDARRAY_FLATTEN_NEXTCARRY_ERRORS::IND_OUT_OF_RANGE)
      }
      else if (j >= 0) {
        tocarry[scan_in_array[thread_id] - 1] = j;
      }
    }
  }
}
