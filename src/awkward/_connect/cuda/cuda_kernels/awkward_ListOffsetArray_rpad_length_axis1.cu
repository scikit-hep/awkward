// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tooffsets, fromoffsets, fromlength, target, tolength, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(fromlength, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_rpad_length_axis1_a", tooffsets.dtype, fromoffsets.dtype, tolength.dtype]))(grid, block, (tooffsets, fromoffsets, fromlength, target, tolength, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_rpad_length_axis1_b", tooffsets.dtype, fromoffsets.dtype, tolength.dtype]))(grid, block, (tooffsets, fromoffsets, fromlength, target, tolength, scan_in_array, invocation_index, err_code))
// out["awkward_ListOffsetArray_rpad_length_axis1_a", {dtype_specializations}] = None
// out["awkward_ListOffsetArray_rpad_length_axis1_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_ListOffsetArray_rpad_length_axis1_a(
    T* tooffsets,
    const C* fromoffsets,
    int64_t fromlength,
    int64_t target,
    U* tolength,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < fromlength) {
      int64_t rangeval = fromoffsets[thread_id + 1] - fromoffsets[thread_id];
      int64_t longer = (target < rangeval) ? rangeval : target;
      scan_in_array[thread_id] = longer;
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_ListOffsetArray_rpad_length_axis1_b(
    T* tooffsets,
    const C* fromoffsets,
    int64_t fromlength,
    int64_t target,
    U* tolength,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    tooffsets[0] = 0;

    *tolength = fromlength > 0 ? scan_in_array[fromlength - 1] : 0;
    if (thread_id < fromlength) {
      tooffsets[thread_id + 1] = (T)scan_in_array[thread_id];
    }
  }
}
