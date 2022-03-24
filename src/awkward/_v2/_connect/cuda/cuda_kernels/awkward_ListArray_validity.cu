// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class LISTARRAY_VALIDITY_ERRORS {
  ERROR_START_STOP,   // message: "start[i] > stop[i]"
  ERROR_START_ZERO,   // message: "start[i] < 0"
  ERROR_STOP_CONTENT  // message: "stop[i] > len(content)"
};

template <typename C, typename T>
__global__ void
awkward_ListArray_validity(const C* starts,
                           const T* stops,
                           int64_t length,
                           int64_t lencontent,
                           uint64_t invocation_index,
                           uint64_t* err_code) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (err_code[0] == NO_ERROR) {
    if (thread_id < length) {
      C start = starts[thread_id];
      T stop = stops[thread_id];
      if (start != stop) {
        if (start > stop) {
          RAISE_ERROR(LISTARRAY_VALIDITY_ERRORS::ERROR_START_STOP)
        }
        if (start < 0) {
          RAISE_ERROR(LISTARRAY_VALIDITY_ERRORS::ERROR_START_ZERO)
        }
        if (stop > lencontent) {
          RAISE_ERROR(LISTARRAY_VALIDITY_ERRORS::ERROR_STOP_CONTENT)
        }
      }
    }
  }
}
