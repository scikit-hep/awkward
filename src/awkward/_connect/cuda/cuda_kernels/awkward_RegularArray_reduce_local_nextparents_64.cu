// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (nextparents, size, length, invocation_index, err_code) = args
//     scan_in_array = cupy.ones(length * size, dtype=cupy.int64)
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_RegularArray_reduce_local_nextparents_64', nextparents.dtype]))(grid, block, (nextparents, size, length, scan_in_array, invocation_index, err_code))
// out["awkward_RegularArray_reduce_local_nextparents_64", {dtype_specializations}] = None
// END PYTHON

template <typename T>
__global__ void
awkward_RegularArray_reduce_local_nextparents_64(
    T* nextparents,
    int64_t size,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) / size;
    int64_t thready_id = (blockIdx.x * blockDim.x + threadIdx.x) % size;
    if (thread_id < length && thready_id < size) {
      nextparents[scan_in_array[thread_id * size + thready_id] - 1] = thread_id;
    }
  }
}
