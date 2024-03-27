// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (lenout, tocarry, fromtags, fromindex, length, which, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_UnionArray_project_a", lenout.dtype, tocarry.dtype, fromtags.dtype, fromindex.dtype]))(grid, block, (lenout, tocarry, fromtags, fromindex, length, which, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_UnionArray_project_b", lenout.dtype, tocarry.dtype, fromtags.dtype, fromindex.dtype]))(grid, block, (lenout, tocarry, fromtags, fromindex, length, which, scan_in_array, invocation_index, err_code))
// out["awkward_UnionArray_project_a", {dtype_specializations}] = None
// out["awkward_UnionArray_project_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_UnionArray_project_a(
    T* lenout,
    C* tocarry,
    const U* fromtags,
    const V* fromindex,
    int64_t length,
    int64_t which,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      if (fromtags[thread_id] == which) {
        scan_in_array[thread_id] = 1;
      }
    }
  }
}

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_UnionArray_project_b(
    T* lenout,
    C* tocarry,
    const U* fromtags,
    const V* fromindex,
    int64_t length,
    int64_t which,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    *lenout = length > 0 ? scan_in_array[length - 1] : 0;
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      if (fromtags[thread_id] == which) {
        tocarry[scan_in_array[thread_id] - 1] = fromindex[thread_id];
      }
    }
  }
}
