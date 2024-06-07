// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toindex, fromoffsets, fromlength, target, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(fromlength, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_rpad_axis1_a", toindex.dtype, fromoffsets.dtype]))(grid, block, (toindex, fromoffsets, fromlength, target, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_rpad_axis1_b", toindex.dtype, fromoffsets.dtype]))(grid, block, (toindex, fromoffsets, fromlength, target, scan_in_array, invocation_index, err_code))
// out["awkward_ListOffsetArray_rpad_axis1_a", {dtype_specializations}] = None
// out["awkward_ListOffsetArray_rpad_axis1_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C>
__global__ void
awkward_ListOffsetArray_rpad_axis1_a(
    T* toindex,
    const C* fromoffsets,
    int64_t fromlength,
    int64_t target,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < fromlength) {
      int64_t rangeval =
          (T)(fromoffsets[thread_id + 1] - fromoffsets[thread_id]);
      scan_in_array[thread_id + 1] = rangeval > target ? rangeval : target;
    }
  }
}

template <typename T, typename C>
__global__ void
awkward_ListOffsetArray_rpad_axis1_b(
    T* toindex,
    const C* fromoffsets,
    int64_t fromlength,
    int64_t target,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < fromlength) {
      int64_t rangeval =
          (T)(fromoffsets[thread_id + 1] - fromoffsets[thread_id]);
      int64_t index = scan_in_array[thread_id];

      for (int64_t j = threadIdx.y; j < rangeval; j += blockDim.y) {
        toindex[index + j] = (T)fromoffsets[thread_id] + j;
      }
      for (int64_t j = rangeval + threadIdx.y; j < target; j += blockDim.y) {
        toindex[index + j] = -1;
      }
    }
  }
}
