// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C>
__global__ void
awkward_index_rpad_and_clip_axis1(
    T* tostarts,
    C* tostops,
    int64_t target,
    int64_t length,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      tostarts[thread_id] = thread_id * target;
      tostops[thread_id] = (thread_id + 1) * target;
    }
  }
}
