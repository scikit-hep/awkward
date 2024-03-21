// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (numnull, fromindex, lenindex, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(lenindex, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_IndexedArray_numnull_a', numnull.dtype, fromindex.dtype]))(grid, block, (numnull, fromindex, lenindex, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_IndexedArray_numnull_b', numnull.dtype, fromindex.dtype]))(grid, block, (numnull, fromindex, lenindex, scan_in_array, invocation_index, err_code))
// out["awkward_IndexedArray_numnull_a", {dtype_specializations}] = None
// out["awkward_IndexedArray_numnull_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C>
__global__ void
awkward_IndexedArray_numnull_a(
    T* numnull,
    const C* fromindex,
    int64_t lenindex,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < lenindex) {
      if (fromindex[thread_id] < 0) {
        scan_in_array[thread_id] = 1;
      }
    }
  }
}

template <typename T, typename C>
__global__ void
awkward_IndexedArray_numnull_b(
    T* numnull,
    const C* fromindex,
    int64_t lenindex,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    *numnull = lenindex > 0 ? scan_in_array[lenindex - 1] : 0;
  }
}
