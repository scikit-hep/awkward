// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (nextcarry, nextparents, outindex, index, parents, length, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_reduce_next_64_a", nextcarry.dtype, nextparents.dtype, outindex.dtype, index.dtype, parents.dtype]))(grid, block, (nextcarry, nextparents, outindex, index, parents, length, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_reduce_next_64_b", nextcarry.dtype, nextparents.dtype, outindex.dtype, index.dtype, parents.dtype]))(grid, block, (nextcarry, nextparents, outindex, index, parents, length, scan_in_array, invocation_index, err_code))
// out["awkward_IndexedArray_reduce_next_64_a", {dtype_specializations}] = None
// out["awkward_IndexedArray_reduce_next_64_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_IndexedArray_reduce_next_64_a(
    T* nextcarry,
    C* nextparents,
    U* outindex,
    const V* index,
    const W* parents,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if (index[thread_id] >= 0) {
        scan_in_array[thread_id] = 1;
      }
    }
  }
}

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_IndexedArray_reduce_next_64_b(
    T* nextcarry,
    C* nextparents,
    U* outindex,
    const V* index,
    const W* parents,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if (index[thread_id] >= 0) {
        nextcarry[scan_in_array[thread_id] - 1] = index[thread_id];
        nextparents[scan_in_array[thread_id] - 1] = parents[thread_id];
        outindex[thread_id] = scan_in_array[thread_id] - 1;
      } else {
        outindex[thread_id] = -1;
      }
    }
  }
}
