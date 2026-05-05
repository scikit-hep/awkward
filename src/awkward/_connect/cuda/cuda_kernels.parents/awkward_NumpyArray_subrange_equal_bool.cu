// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tmpptr, fromstarts, fromstops, length, toequal, invocation_index, err_code) = args
//     if length > 1:
//         scan_in_array = cupy.full((length - 1) * (length - 2), cupy.array(0), dtype=cupy.int64)
//     else:
//         scan_in_array = cupy.full(0, cupy.array(0), dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_NumpyArray_subrange_equal_bool", bool_, fromstarts.dtype, fromstops.dtype, bool_]))(grid, block, (tmpptr, fromstarts, fromstops, length, toequal, scan_in_array, invocation_index, err_code))
//     toequal[0] = cupy.any(scan_in_array == True)
// out["awkward_NumpyArray_subrange_equal_bool", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_NumpyArray_subrange_equal_bool(
    T* tmpptr,
    const C* fromstarts,
    const U* fromstops,
    int64_t length,
    V* toequal,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    bool differ = true;
    int64_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) / (length - 1);
    int64_t thready_id = (blockIdx.x * blockDim.x + threadIdx.x) % (length - 1);
    if (thread_id < length - 1 && thready_id < length - 1) {
      int64_t leftlen = fromstops[thread_id] - fromstarts[thread_id];
      if (thready_id > thread_id) {
        int64_t rightlen = fromstops[thready_id] - fromstarts[thready_id];
        if (leftlen == rightlen) {
          differ = false;
          for (int64_t j = threadIdx.y; j < leftlen; j += blockDim.y) {
            if ((tmpptr[fromstarts[thread_id] + j] != 0) != (tmpptr[fromstarts[thready_id] + j] != 0)) {
              differ = true;
              break;
            }
          }
        }
        int64_t idx = thread_id * (length - 3) + thready_id - 1;
        scan_in_array[idx] = !differ;
      }
    }
  }
}
