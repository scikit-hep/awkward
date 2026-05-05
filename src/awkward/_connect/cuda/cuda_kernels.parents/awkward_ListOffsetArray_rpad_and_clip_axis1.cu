// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C>
__global__ void
awkward_ListOffsetArray_rpad_and_clip_axis1(
    T* toindex,
    const C* fromoffsets,
    int64_t length,
    int64_t target,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) / target;
    int64_t thready_id = (blockIdx.x * blockDim.x + threadIdx.x) % target;

    if (thread_id < length) {
      int64_t rangeval =
          (T)(fromoffsets[thread_id + 1] - fromoffsets[thread_id]);
      int64_t shorter = (target < rangeval) ? target : rangeval;

      if (thready_id < shorter) {
        toindex[thread_id * target + thready_id] =
            (T)fromoffsets[thread_id] + thready_id;
      } else if (thready_id >= shorter && thready_id < target) {
        toindex[thready_id * target + thready_id] = -1;
      }
    }
  }
}
