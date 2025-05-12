// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (totallen, tooffsets, n, replacement, starts, stops, length, invocation_index, err_code) = args
//     scan_out = cupy.zeros(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_combinations_length_a", totallen.dtype, tooffsets.dtype, starts.dtype, stops.dtype]))(grid, block, (totallen, tooffsets, n, replacement, starts, stops, length, scan_out, invocation_index, err_code))
//     cupy.cumsum(scan_out, out=scan_out)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_combinations_length_b", totallen.dtype, tooffsets.dtype, starts.dtype, stops.dtype]))(grid, block, (totallen, tooffsets, n, replacement, starts, stops, length, scan_out, invocation_index, err_code))
// out["awkward_ListArray_combinations_length_a", {dtype_specializations}] = None
// out["awkward_ListArray_combinations_length_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_ListArray_combinations_length_a(
    T* totallen,
    C* tooffsets,
    int64_t n,
    bool replacement,
    const U* starts,
    const V* stops,
    int64_t length,
    int64_t* scan_out,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] != NO_ERROR) {
    return;
  }

  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id >= length) {
    return;
  }

  int64_t size = stops[thread_id] - starts[thread_id];
  int64_t combinationslen = 0;

  if (replacement) {
    size += (n - 1);
  }

  if (n > size) {
    combinationslen = 0;
  }
  else if (n == size) {
    combinationslen = 1;
  }
  else {
    // Choose the smaller of n and size - n for fewer multiplications
    int64_t k = (n * 2 > size) ? (size - n) : n;

    combinationslen = 1;
    for (int64_t j = 1; j <= k; ++j) {
      combinationslen = (combinationslen * (size - j + 1)) / j;
    }
  }

  scan_out[thread_id] = combinationslen;
}

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_ListArray_combinations_length_b(
    T* totallen,
    C* tooffsets,
    int64_t n,
    bool replacement,
    const U* starts,
    const V* stops,
    int64_t length,
    int64_t* scan_out,
    uint64_t invocation_index,
    uint64_t* err_code) {

  if (err_code[0] != NO_ERROR) {
    return;
  }

  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  // Let a single thread handle totallen and tooffsets[0]
  if (thread_id == 0) {
    *totallen = (length > 0) ? scan_out[length - 1] : 0;
    tooffsets[0] = 0;
  }

  // Copy scan_out values into tooffsets (shifted by 1)
  if (thread_id < length) {
    tooffsets[thread_id + 1] = scan_out[thread_id];
  }
}
