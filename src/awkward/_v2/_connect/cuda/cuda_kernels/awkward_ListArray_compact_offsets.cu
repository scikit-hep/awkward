// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class LISTARRAY_COMPACT_OFFSETS_ERRORS {
  ERROR_START_STOP,  // message: "start[i] > stop[i]"
};

// BEGIN PYTHON
// def f(grid, block, args):
//     (tooffsets, fromstarts, fromstops, length, invocation_index, err_code) = args
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_compact_offsets_a", tooffsets.dtype, fromstarts.dtype, fromstops.dtype]))(grid, block, (tooffsets, fromstarts, fromstops, length, invocation_index, err_code))
//     tooffsets = inclusive_scan(grid, block, (tooffsets, invocation_index, err_code))
// out["awkward_ListArray_compact_offsets_a", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_ListArray_compact_offsets_a(T* tooffsets,
                                    const C* fromstarts,
                                    const U* fromstops,
                                    int64_t length,
                                    uint64_t invocation_index,
                                    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    tooffsets[0] = 0;
    if (thread_id < length) {
      C start = fromstarts[thread_id];
      U stop = fromstops[thread_id];
      if (stop < start) {
        RAISE_ERROR(LISTARRAY_COMPACT_OFFSETS_ERRORS::ERROR_START_STOP)
      }
      tooffsets[thread_id + 1] = (stop - start);
    }
  }
}
