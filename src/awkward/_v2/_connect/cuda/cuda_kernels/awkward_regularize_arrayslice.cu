// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class REGULARIZE_ARRYSLICE {
  IND_OUT_OF_RANGE,  // message: "index out of range"
};

template <typename T>
__global__ void
awkward_regularize_arrayslice(T* flatheadptr,
                              int64_t lenflathead,
                              int64_t length,
                              uint64_t invocation_index,
                              uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < lenflathead) {
      if (flatheadptr[thread_id] < -length ||
          flatheadptr[thread_id] >= length) {
        RAISE_ERROR(REGULARIZE_ARRYSLICE::IND_OUT_OF_RANGE)
      }
      if (flatheadptr[thread_id] < 0) {
        flatheadptr[thread_id] += length;
      }
    }
  }
}
