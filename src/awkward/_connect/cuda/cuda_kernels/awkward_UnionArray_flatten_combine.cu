// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (totags, toindex, tooffsets, fromtags, fromindex, length, offsetsraws, invocation_index, err_code) = args
//     scan_in_array_tooffsets = cupy.zeros(length + 1, dtype=cupy.int64)
//     scan_in_array_k = cupy.ones(length*length, dtype=cupy.int64)
//     scan_in_array_k = cupy.cumsum(scan_in_array_k)
//     print(offsetsraws)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_UnionArray_flatten_combine_a', totags.dtype, toindex.dtype, tooffsets.dtype, fromtags.dtype, fromindex.dtype, offsetsraws.dtype]))(grid, block, (totags, toindex, tooffsets, fromtags, fromindex, length, offsetsraws, scan_in_array_tooffsets, scan_in_array_k, invocation_index, err_code))
//     scan_in_array_tooffsets = cupy.cumsum(scan_in_array_tooffsets)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_UnionArray_flatten_combine_b', totags.dtype, toindex.dtype, tooffsets.dtype, fromtags.dtype, fromindex.dtype, offsetsraws.dtype]))(grid, block, (totags, toindex, tooffsets, fromtags, fromindex, length, offsetsraws, scan_in_array_tooffsets, scan_in_array_k, invocation_index, err_code))
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
    int64_t* scan_in_array_k,
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
      for (int64_t j = start;  j < stop;  j++) {
        totags[scan_in_array_k[j] - 1] = tag;
        toindex[scan_in_array_k[j] - 1] = j;
      }
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
    int64_t* scan_in_array_k,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id <= length) {
      tooffsets[thread_id] = scan_in_array_tooffsets[thread_id];
    }
  }
}

// does not take 2d array as input
