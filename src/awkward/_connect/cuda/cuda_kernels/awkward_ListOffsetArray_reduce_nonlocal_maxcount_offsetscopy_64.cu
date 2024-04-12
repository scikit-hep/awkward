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

template <typename T, typename C, typename U>
__global__ void
awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64_a(
    T* maxcount,
    C* offsetscopy,
    const U* offsets,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    offsetscopy[0] = offsets[0];

    if (thread_id < length) {
      if(thread_id == 0) {
        scan_in_array[0] = 0;
      }
      int64_t count = (offsets[thread_id + 1] - offsets[thread_id]);
      scan_in_array[thread_id + 1] = count;
      offsetscopy[thread_id + 1] = offsets[thread_id + 1];
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64_b(
    T* maxcount,
    C* offsetscopy,
    const U* offsets,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    *maxcount = length > 0 ? scan_in_array[0]: 0;
  }
}
