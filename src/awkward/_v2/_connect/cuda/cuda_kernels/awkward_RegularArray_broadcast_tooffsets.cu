// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class REGULARARRAY_BROADCAST_TOOFFSETS_ERRRORS {
  OFFSETS_MON_INCR,  // message: "broadcast's offsets must be monotonically increasing"
  BROAD_NEST_LIST  // message: "cannot broadcast nested list"
};

template <typename T>
__global__ void
awkward_RegularArray_broadcast_tooffsets(const T* fromoffsets,
                                         int64_t offsetslength,
                                         int64_t size,
                                         uint64_t invocation_index,
                                         uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < (offsetslength - 1)) {
      int64_t count =
          (int64_t)(fromoffsets[(thread_id + 1)] - fromoffsets[thread_id]);
      if (count < 0) {
        RAISE_ERROR(REGULARARRAY_BROADCAST_TOOFFSETS_ERRRORS::OFFSETS_MON_INCR)
      }
      if (size != count) {
        RAISE_ERROR(REGULARARRAY_BROADCAST_TOOFFSETS_ERRRORS::BROAD_NEST_LIST)
      }
    }
  }
}
