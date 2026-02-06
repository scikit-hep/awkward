// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (maxcount, offsetscopy, offsets, length, invocation_index, err_code) = args
//     scan_in_array = cupy.empty(length + 1, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64_a", maxcount.dtype, offsetscopy.dtype, offsets.dtype]))(grid, block, (maxcount, offsetscopy, offsets, length, scan_in_array, invocation_index, err_code))
//     if length > 0:
//         scan_in_array[0] = cupy.max(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64_b", maxcount.dtype, offsetscopy.dtype, offsets.dtype]))(grid, block, (maxcount, offsetscopy, offsets, length, scan_in_array, invocation_index, err_code))
// out["awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64_a", {dtype_specializations}] = None
// out["awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64_b", {dtype_specializations}] = None
// END PYTHON

template <typename T_maxcount, typename T_offsetscopy, typename T_offsets>
__global__ void
awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64_a(
    T_maxcount* maxcount,
    T_offsetscopy* offsetscopy,
    const T_offsets* offsets,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id == 0) {
      offsetscopy[0] = offsets[0];
      scan_in_array[0] = 0;
    }

    if (thread_id < length) {
      T_offsets count = offsets[thread_id + 1] - offsets[thread_id];
      scan_in_array[thread_id + 1] = (int64_t)count;
      offsetscopy[thread_id + 1] = offsets[thread_id + 1];
    }
  }
}

template <typename T_maxcount, typename T_offsetscopy, typename T_offsets>
__global__ void
awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64_b(
    T_maxcount* maxcount,
    T_offsetscopy* offsetscopy,
    const T_offsets* offsets,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    *maxcount = (T_maxcount)(length > 0 ? scan_in_array[0] : 0);
  }
}
