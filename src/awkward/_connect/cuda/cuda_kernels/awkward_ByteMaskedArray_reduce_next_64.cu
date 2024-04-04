// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (nextcarry, nextparents, outindex, mask, parents, length, validwhen, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_ByteMaskedArray_reduce_next_64_a', nextcarry.dtype, nextparents.dtype, outindex.dtype, mask.dtype, parents.dtype]))(grid, block, (nextcarry, nextparents, outindex, mask, parents, length, validwhen, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_ByteMaskedArray_reduce_next_64_b', nextcarry.dtype, nextparents.dtype, outindex.dtype, mask.dtype, parents.dtype]))(grid, block, (nextcarry, nextparents, outindex, mask, parents, length, validwhen, scan_in_array, invocation_index, err_code))
// out["awkward_ByteMaskedArray_reduce_next_64_a", {dtype_specializations}] = None
// out["awkward_ByteMaskedArray_reduce_next_64_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_ByteMaskedArray_reduce_next_64_a(
    T* nextcarry,
    C* nextparents,
    U* outindex,
    const V* mask,
    const W* parents,
    int64_t length,
    bool validwhen,
    int64_t* scan_in_array,
    uint64_t* invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if ((mask[thread_id] != 0) == validwhen) {
        scan_in_array[thread_id] = 1;
      }
    }
  }
}

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_ByteMaskedArray_reduce_next_64_b(
    T* nextcarry,
    C* nextparents,
    U* outindex,
    const V* mask,
    const W* parents,
    int64_t length,
    bool validwhen,
    int64_t* scan_in_array,
    uint64_t* invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if ((mask[thread_id] != 0) == validwhen) {
        nextcarry[scan_in_array[thread_id] - 1] = thread_id;
        nextparents[scan_in_array[thread_id] - 1] = parents[thread_id];
        outindex[thread_id] = scan_in_array[thread_id] - 1;
      } else {
        outindex[thread_id] = -1;
      }
    }
  }
}
