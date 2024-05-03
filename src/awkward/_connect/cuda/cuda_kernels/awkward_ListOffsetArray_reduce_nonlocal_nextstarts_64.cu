// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (nextstarts, nextparents, nextlen, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(nextlen, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64_a", nextstarts.dtype, nextparents.dtype]))(grid, block, (nextstarts, nextparents, nextlen, scan_in_array, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64_b", nextstarts.dtype, nextparents.dtype]))(grid, block, (nextstarts, nextparents, nextlen, scan_in_array, invocation_index, err_code))
// out["awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64_a", {dtype_specializations}] = None
// out["awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C>
__global__ void
awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64_a(
    T* nextstarts,
    const C* nextparents,
    int64_t nextlen,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < nextlen) {
      if (thread_id == 0) {
        scan_in_array[0] = -1;
      }
      scan_in_array[thread_id + 1] = nextparents[thread_id];
    }
  }
}

template <typename T, typename C>
__global__ void
awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64_b(
    T* nextstarts,
    const C* nextparents,
    int64_t nextlen,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < nextlen) {
      if (nextparents[thread_id] != scan_in_array[thread_id]) {
        nextstarts[nextparents[thread_id]] = thread_id;
      }
    }
  }
}
