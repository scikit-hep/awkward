// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (nextcarry, nextparents, parents, size, length, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(length * size, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_RegularArray_reduce_nonlocal_preparenext_a', nextcarry.dtype, nextparents.dtype, parents.dtype]))(grid, block, (nextcarry, nextparents, parents, size, length, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_RegularArray_reduce_nonlocal_preparenext_b', nextcarry.dtype, nextparents.dtype, parents.dtype]))(grid, block, (nextcarry, nextparents, parents, size, length, scan_in_array, invocation_index, err_code))
// out["awkward_RegularArray_reduce_nonlocal_preparenext_a", {dtype_specializations}] = None
// out["awkward_RegularArray_reduce_nonlocal_preparenext_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_RegularArray_reduce_nonlocal_preparenext_a(
    T* nextcarry,
    C* nextparents,
    const U* parents,
    int64_t size,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
if (err_code[0] == NO_ERROR) {
    int64_t thready_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t len = length * size;
    if (thready_id < len) {
      scan_in_array[thready_id] = 1;
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_RegularArray_reduce_nonlocal_preparenext_b(
    T* nextcarry,
    C* nextparents,
    const U* parents,
    int64_t size,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thready_id = (blockIdx.x * blockDim.x + threadIdx.x) / length;
    int64_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) % length;
    if (thread_id < length && thready_id < size) {
      nextcarry[scan_in_array[thready_id * length + thread_id] - 1] = thread_id * size + thready_id;
      nextparents[scan_in_array[thready_id * length + thread_id] - 1] = parents[thread_id] * size + thready_id;
    }
  }
}
