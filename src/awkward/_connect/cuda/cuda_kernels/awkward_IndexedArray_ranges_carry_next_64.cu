// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (index, fromstarts, fromstops, length, tocarry, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros_like(index, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_ranges_carry_next_64_a", index.dtype, fromstarts.dtype, fromstops.dtype, tocarry.dtype]))(grid, block, (index, fromstarts, fromstops, length, tocarry, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_ranges_carry_next_64_b", index.dtype, fromstarts.dtype, fromstops.dtype, tocarry.dtype]))(grid, block, (index, fromstarts, fromstops, length, tocarry, scan_in_array, invocation_index, err_code))
// out["awkward_IndexedArray_ranges_carry_next_64_a", {dtype_specializations}] = None
// out["awkward_IndexedArray_ranges_carry_next_64_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_IndexedArray_ranges_carry_next_64_a(
    T* index,
    const C* fromstarts,
    const U* fromstops,
    int64_t length,
    V* tocarry,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = 0;

    if (thread_id < length) {
      stride = fromstops[thread_id] - fromstarts[thread_id];
      for (int64_t j = 0; j < stride; j++) {
        if (!(index[fromstarts[thread_id] + j] < 0)) {
          scan_in_array[fromstarts[thread_id] + j] = 1;
        }
      }
    }
  }
}

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_IndexedArray_ranges_carry_next_64_b(
    T* index,
    const C* fromstarts,
    const U* fromstops,
    int64_t length,
    V* tocarry,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = 0;

    if (thread_id < length) {
      stride = fromstops[thread_id] - fromstarts[thread_id];
      for (int64_t j = 0; j < stride; j++) {
        if (!(index[fromstarts[thread_id] + j] < 0)) {
          tocarry[scan_in_array[fromstarts[thread_id] + j] - 1] = index[fromstarts[thread_id] + j];
        }
      }
    }
  }
}
