// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class LISTARRAY_COMPACT_OFFSETS_ERRORS {
  ERROR_START_STOP,  // message: "start[i] > stop[i]"
};

template <typename T, typename C, typename U>
__global__ void
awkward_ListArray_compact_offsets(T* tooffsets,
                                  const C* fromstarts,
                                  const U* fromstops,
                                  int64_t length,
                                  uint64_t invocation_index,
                                  uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      if (thread_id == 0) {
        tooffsets[thread_id] = 0;
      }
      C start = fromstarts[thread_id];
      U stop = fromstops[thread_id];
      if (stop < start) {
        RAISE_ERROR(LISTARRAY_COMPACT_OFFSETS_ERRORS::ERROR_START_STOP)
      }
      tooffsets[thread_id + 1] = tooffsets[thread_id] + (stop - start);
    }
  }
}
