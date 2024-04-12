// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toindex, frommask, length, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_a", toindex.dtype, frommask.dtype]))(grid, block, (toindex, frommask, length, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_b", toindex.dtype, frommask.dtype]))(grid, block, (toindex, frommask, length, scan_in_array, invocation_index, err_code))
// out["awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_a", {dtype_specializations}] = None
// out["awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C>
__global__ void
awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_a(
    T* toindex,
    const C* frommask,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if (!frommask[thread_id]) {
        scan_in_array[thread_id] = 1;
      }
    }
  }
}

template <typename T, typename C>
__global__ void
awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_b(
    T* toindex,
    const C* frommask,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if (frommask[thread_id]) {
        toindex[thread_id] = -1;
      } else {
        toindex[thread_id] = scan_in_array[thread_id] - 1;
      }
    }
  }
}
