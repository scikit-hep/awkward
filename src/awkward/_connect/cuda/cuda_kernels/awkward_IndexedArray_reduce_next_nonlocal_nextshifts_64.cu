// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (nextshifts, index, length, invocation_index, err_code) = args
//     scan_in_array_k = cupy.zeros(length, dtype=cupy.int64)
//     scan_in_array_nullsum = cupy.zeros(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64_a", nextshifts.dtype, index.dtype]))(grid, block, (nextshifts, index, length, scan_in_array_k, scan_in_array_nullsum, invocation_index, err_code))
//     scan_in_array_k = cupy.cumsum(scan_in_array_k)
//     scan_in_array_nullsum = cupy.cumsum(scan_in_array_nullsum)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64_b", nextshifts.dtype, index.dtype]))(grid, block, (nextshifts, index, length, scan_in_array_k, scan_in_array_nullsum, invocation_index, err_code))
// out["awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64_a", {dtype_specializations}] = None
// out["awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C>
__global__ void
awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64_a(
    T* nextshifts,
    const C* index,
    int64_t length,
    int64_t* scan_in_array_k,
    int64_t* scan_in_array_nullsum,
    uint64_t invocation_code,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if (index[thread_id] >= 0) {
        scan_in_array_k[thread_id] = 1;
      } else {
        scan_in_array_nullsum[thread_id] = 1;
      }
    }
  }
}

template <typename T, typename C>
__global__ void
awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64_b(
    T* nextshifts,
    const C* index,
    int64_t length,
    int64_t* scan_in_array_k,
    int64_t* scan_in_array_nullsum,
    uint64_t invocation_code,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if (index[thread_id] >= 0) {
        nextshifts[scan_in_array_k[thread_id] - 1] = scan_in_array_nullsum[thread_id];
      }
    }
  }
}
