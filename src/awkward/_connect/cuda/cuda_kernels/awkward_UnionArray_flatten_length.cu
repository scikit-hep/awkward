// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (total_length, fromtags, fromindex, length, offsetsraws, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_UnionArray_flatten_length_a', total_length.dtype, fromtags.dtype, fromindex.dtype, offsetsraws[0].dtype]))(grid, block, (total_length, fromtags, fromindex, length, offsetsraws, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_UnionArray_flatten_length_b', total_length.dtype, fromtags.dtype, fromindex.dtype, offsetsraws[0].dtype]))(grid, block, (total_length, fromtags, fromindex, length, offsetsraws, scan_in_array, invocation_index, err_code))
// out["awkward_UnionArray_flatten_length_a", {dtype_specializations}] = None
// out["awkward_UnionArray_flatten_length_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_UnionArray_flatten_length_a(
    T* total_length,
    const C* fromtags,
    const U* fromindex,
    int64_t length,
    V** offsetsraws,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      C tag = fromtags[thread_id];
      U idx = fromindex[thread_id];
      V start = offsetsraws[tag][idx];
      V stop = offsetsraws[tag][idx + 1];
      scan_in_array[thread_id] = stop - start;
    }
  }
}

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_UnionArray_flatten_length_b(
    T* total_length,
    const C* fromtags,
    const U* fromindex,
    int64_t length,
    V** offsetsraws,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    *total_length = length > 0 ? scan_in_array[length - 1] : 0;
  }
}
