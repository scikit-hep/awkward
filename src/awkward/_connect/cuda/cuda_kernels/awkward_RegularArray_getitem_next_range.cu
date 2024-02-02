// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T>
__global__ void
awkward_RegularArray_getitem_next_range(
    T* tocarry,
    int64_t regular_start,
    int64_t step,
    int64_t length,
    int64_t size,
    int64_t nextsize,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) / nextsize;
    int64_t thready_id = (blockIdx.x * blockDim.x + threadIdx.x) % nextsize;

    if (thread_id < length) {
      tocarry[(thread_id * nextsize) + thready_id] =
          ((thread_id * size) + regular_start) + (thready_id * step);
    }
  }
}
