// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (carrylength, fromstarts, fromstops, lenstarts, start, stop, step, invocation_index, err_code) = args
//     scan_in_array_carrylength = cupy.zeros(lenstarts, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_getitem_next_range_carrylength_a", carrylength.dtype, fromstarts.dtype, fromstops.dtype]))(grid, block, (carrylength, fromstarts, fromstops, lenstarts, start, stop, step, scan_in_array_carrylength, invocation_index, err_code))
//     scan_in_array_carrylength = cupy.cumsum(scan_in_array_carrylength)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_getitem_next_range_carrylength_b", carrylength.dtype, fromstarts.dtype, fromstops.dtype]))(grid, block, (carrylength, fromstarts, fromstops, lenstarts, start, stop, step, scan_in_array_carrylength, invocation_index, err_code))
// out["awkward_ListArray_getitem_next_range_carrylength_a", {dtype_specializations}] = None
// out["awkward_ListArray_getitem_next_range_carrylength_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_ListArray_getitem_next_range_carrylength_a(
    T* carrylength,
    const C* fromstarts,
    const U* fromstops,
    int64_t lenstarts,
    int64_t start,
    int64_t stop,
    int64_t step,
    int64_t* scan_in_array_carrylength,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < lenstarts) {
      int64_t length = fromstops[thread_id] - fromstarts[thread_id];
      int64_t regular_start = start;
      int64_t regular_stop = stop;
      awkward_regularize_rangeslice(&regular_start, &regular_stop, step > 0,
                                    start != kSliceNone, stop != kSliceNone,
                                    length);
      int64_t carrylen = 0;
      if (step > 0) {
        for (int64_t j = regular_start + step * threadIdx.y;  j < regular_stop;  j += step * blockDim.y) {
            carrylen += 1;
        }
      }
      else {
        for (int64_t j = regular_start - step * threadIdx.y;  j > regular_stop;  j += step * blockDim.y) {
            carrylen += 1;
        }
      }
      scan_in_array_carrylength[thread_id] = carrylen;
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_ListArray_getitem_next_range_carrylength_b(
    T* carrylength,
    const C* fromstarts,
    const U* fromstops,
    int64_t lenstarts,
    int64_t start,
    int64_t stop,
    int64_t step,
    int64_t* scan_in_array_carrylength,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    *carrylength = lenstarts > 0 ? scan_in_array_carrylength[lenstarts - 1] : 0;
  }
}
