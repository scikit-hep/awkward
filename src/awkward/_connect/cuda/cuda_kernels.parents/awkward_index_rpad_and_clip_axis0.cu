// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T>
__global__ void
awkward_index_rpad_and_clip_axis0(
    T* toindex,
    int64_t target,
    int64_t length,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      int64_t shorter = (target < length ? target : length);
      if (thread_id < shorter) {
        toindex[thread_id] = thread_id;
      } else if (thread_id >= shorter && thread_id < target) {
        toindex[thread_id] = -1;
      }
    }
  }
}
