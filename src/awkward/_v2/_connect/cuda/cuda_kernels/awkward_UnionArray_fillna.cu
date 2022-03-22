// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename A>
__global__ void
awkward_UnionArray_fillna(int64_t* toindex,
                          const A* fromindex,
                          int64_t length,
                          uint64_t invocation_index,
                          uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      if ((fromindex[thread_id] >= 0)) {
        toindex[thread_id] = fromindex[thread_id];
      } else {
        toindex[thread_id] = 0;
      }
    }
  }
}
