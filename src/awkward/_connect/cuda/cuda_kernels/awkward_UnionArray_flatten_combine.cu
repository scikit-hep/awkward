// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (totags, toindex, tooffsets, fromtags, fromindex, length, offsetsraws, invocation_index, err_code) = args
//     scan_in_array_tooffsets = cupy.zeros(length + 1, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_UnionArray_flatten_combine_a', totags.dtype, toindex.dtype, tooffsets.dtype, fromtags.dtype, fromindex.dtype, offsetsraws[0].dtype]))(grid, block, (totags, toindex, tooffsets, fromtags, fromindex, length, offsetsraws, scan_in_array_tooffsets, invocation_index, err_code))
//     scan_in_array_tooffsets = cupy.cumsum(scan_in_array_tooffsets)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_UnionArray_flatten_combine_b', totags.dtype, toindex.dtype, tooffsets.dtype, fromtags.dtype, fromindex.dtype, offsetsraws[0].dtype]))(grid, block, (totags, toindex, tooffsets, fromtags, fromindex, length, offsetsraws, scan_in_array_tooffsets, invocation_index, err_code))
// out["awkward_UnionArray_flatten_combine_a", {dtype_specializations}] = None
// out["awkward_UnionArray_flatten_combine_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V, typename W, typename X>
__global__ void
awkward_UnionArray_flatten_combine_a(
    T* totags,
    C* toindex,
    U* tooffsets,
    const V* fromtags,
    const W* fromindex,
    int64_t length,
    X** offsetsraws,
    int64_t* scan_in_array_tooffsets,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      V tag = fromtags[thread_id];
      W idx = fromindex[thread_id];
      X start = offsetsraws[tag][idx];
      X stop = offsetsraws[tag][idx + 1];
      scan_in_array_tooffsets[thread_id + 1] = stop - start;
    }
  }
}

template <typename T, typename C, typename U, typename V, typename W, typename X>
__global__ void
awkward_UnionArray_flatten_combine_b(
    T* totags,
    C* toindex,
    U* tooffsets,
    const V* fromtags,
    const W* fromindex,
    int64_t length,
    X** offsetsraws,
    int64_t* scan_in_array_tooffsets,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      V tag = fromtags[thread_id];
      W idx = fromindex[thread_id];
      X start = offsetsraws[tag][idx];
      X stop = offsetsraws[tag][idx + 1];
      int64_t k = scan_in_array_tooffsets[thread_id];
      for (int64_t j = start;  j < stop;  j++) {
        totags[k] = tag;
        toindex[k] = j;
        k++;
      }
      tooffsets[thread_id] = scan_in_array_tooffsets[thread_id];
    }
    tooffsets[length] = scan_in_array_tooffsets[length];
  }
}
