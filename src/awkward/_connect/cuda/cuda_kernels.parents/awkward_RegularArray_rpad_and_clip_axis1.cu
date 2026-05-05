// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T>
__global__ void
awkward_RegularArray_rpad_and_clip_axis1(
    T* toindex,
    int64_t target,
    int64_t size,
    int64_t length,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) / target;
    int64_t thready_id = (blockIdx.x * blockDim.x + threadIdx.x) % target;

    if (thread_id < length) {
      int64_t shorter = (target < size ? target : size);
      if (thready_id < shorter) {
        toindex[thread_id*target + thready_id] = thread_id*size + thready_id;
      } else if (thready_id >= shorter && thready_id < target) {
        toindex[(thread_id * target) + thready_id] = -1;
      }
    }
  }
}
