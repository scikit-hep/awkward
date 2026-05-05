// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tooffsets, tocarry, fromstarts, fromstops, lenstarts, start, stop, step, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(lenstarts + 1, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_getitem_next_range_a", tooffsets.dtype, tocarry.dtype, fromstarts.dtype, fromstops.dtype]))(grid, block, (tooffsets, tocarry, fromstarts, fromstops, lenstarts, start, stop, step, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_getitem_next_range_b", tooffsets.dtype, tocarry.dtype, fromstarts.dtype, fromstops.dtype]))(grid, block, (tooffsets, tocarry, fromstarts, fromstops, lenstarts, start, stop, step, scan_in_array, invocation_index, err_code))
// out["awkward_ListArray_getitem_next_range_a", {dtype_specializations}] = None
// out["awkward_ListArray_getitem_next_range_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_ListArray_getitem_next_range_a(
    T* tooffsets,
    C* tocarry,
    const U* fromstarts,
    const V* fromstops,
    int64_t lenstarts,
    int64_t start,
    int64_t stop,
    int64_t step,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < lenstarts) {
      scan_in_array[0] = 0;
      int64_t length = fromstops[thread_id] - fromstarts[thread_id];
      int64_t regular_start = start;
      int64_t regular_stop = stop;
      awkward_regularize_rangeslice(&regular_start, &regular_stop, step > 0,
                                    start != kSliceNone, stop != kSliceNone,
                                    length);

      if (step != 0) {
        scan_in_array[thread_id + 1] = ceil((float)(regular_stop - regular_start) / step);
      }
      else {
        scan_in_array[thread_id + 1] = 0;
      }
    }
  }
}

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_ListArray_getitem_next_range_b(
    T* tooffsets,
    C* tocarry,
    const U* fromstarts,
    const V* fromstops,
    int64_t lenstarts,
    int64_t start,
    int64_t stop,
    int64_t step,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    tooffsets[0] = 0;
    int64_t k = 0;
    if (thread_id < lenstarts) {
      int64_t length = fromstops[thread_id] - fromstarts[thread_id];
      int64_t regular_start = start;
      int64_t regular_stop = stop;
      awkward_regularize_rangeslice(&regular_start, &regular_stop, step > 0,
                                    start != kSliceNone, stop != kSliceNone,
                                    length);

      if (step > 0) {
        for (int64_t j = regular_start + step * threadIdx.y;  j < regular_stop;  j += step * blockDim.y) {
          tocarry[scan_in_array[thread_id] + k] = fromstarts[thread_id] + j;
          k++;
        }
      }
      else {
        for (int64_t j = regular_start - step * threadIdx.y;  j > regular_stop;  j += step * blockDim.y) {
          tocarry[scan_in_array[thread_id] + k] = fromstarts[thread_id] + j;
          k++;
        }
      }
      tooffsets[thread_id + 1] = (T)scan_in_array[thread_id + 1];
    }
  }
}
