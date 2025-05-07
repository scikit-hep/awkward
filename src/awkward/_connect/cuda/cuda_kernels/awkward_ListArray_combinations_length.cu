// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (totallen, tooffsets, n, replacement, starts, stops, length, invocation_index, err_code) = args
//     scan_out = cupy.zeros(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_combinations_length_a", totallen.dtype, tooffsets.dtype, starts.dtype, stops.dtype]))(grid, block, (totallen, tooffsets, n, replacement, starts, stops, length, scan_out, err_code))
//     cupy.cumsum(scan_out, out=scan_out) # in-place cumsum
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_combinations_length_b", totallen.dtype, tooffsets.dtype, starts.dtype, stops.dtype]))(grid, block, (totallen, tooffsets, n, replacement, starts, stops, length, scan_out, err_code))
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
    uint64_t* err_code) {

  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= length || err_code[0] != NO_ERROR) return;

  int64_t size = (int64_t)(stops[tid] - starts[tid]);
  if (replacement) {
    size += (n - 1);
  }

  if (size < n) {
    scan_out[tid] = 0;
    return;
  }

  if (size == n) {
    scan_out[tid] = 1;
    return;
  }

  int64_t k = n;
  // leverage symmetry: C(size, k) == C(size, size-k)
  if (k * 2 > size) {
    k = size - k;
  }

  int64_t combinationslen = 1;
  for (int64_t j = 1; j <= k; ++j) {
    combinationslen = (combinationslen * (size - k + j)) / j;
  }

  scan_out[tid] = combinationslen;
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
    uint64_t* err_code) {

  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= length || err_code[0] != NO_ERROR) return;

  if (tid == 0) {
    *totallen = (length > 0) ? scan_out[length - 1] : 0;
    tooffsets[0] = 0;
  }

  tooffsets[tid + 1] = scan_out[tid];
}
