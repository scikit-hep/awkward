// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (totallen, tooffsets, n, replacement, starts, stops, length, invocation_index, err_code) = args
//     scan_in_array_totallen = cupy.zeros(length, dtype=cupy.int64)
//     scan_in_array_tooffsets = cupy.zeros(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_combinations_length_a", totallen.dtype, tooffsets.dtype, starts.dtype, stops.dtype]))(grid, block, (totallen, tooffsets, n, replacement, starts, stops, length, scan_in_array_totallen, scan_in_array_tooffsets, invocation_index, err_code))
//     scan_in_array_totallen = cupy.cumsum(scan_in_array_totallen)
//     scan_in_array_tooffsets = cupy.cumsum(scan_in_array_tooffsets)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_combinations_length_b", totallen.dtype, tooffsets.dtype, starts.dtype, stops.dtype]))(grid, block, (totallen, tooffsets, n, replacement, starts, stops, length, scan_in_array_totallen, scan_in_array_tooffsets,  invocation_index, err_code))
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
    int64_t* scan_in_array_totallen,
    int64_t* scan_in_array_tooffsets,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      int64_t size = (int64_t)(stops[thread_id] - starts[thread_id]);
      if (replacement) {
        size += (n - 1);
      }
      int64_t thisn = n;
      int64_t combinationslen;
      if (thisn > size) {
        combinationslen = 0;
      }
      else if (thisn == size) {
        combinationslen = 1;
      }
      else {
        if (thisn * 2 > size) {
          thisn = size - thisn;
        }
        combinationslen = size;
        for (int64_t j = 2;  j <= thisn;  j++) {
          combinationslen *= (size - j + 1);
          combinationslen /= j;
        }
      }
      scan_in_array_totallen[thread_id] = combinationslen;
      scan_in_array_tooffsets[thread_id] = combinationslen;
    }
  }
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
    int64_t* scan_in_array_totallen,
    int64_t* scan_in_array_tooffsets,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    *totallen = length > 0 ? scan_in_array_totallen[length - 1] : 0;
    tooffsets[0] = 0;

    if (thread_id < length) {
      tooffsets[thread_id + 1] = scan_in_array_tooffsets[thread_id];
    }
  }
}
