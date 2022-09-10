// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C>
__global__ void
awkward_NumpyArray_getitem_next_range(T* nextcarryptr,
                                      const C* carryptr,
                                      int64_t lencarry,
                                      int64_t lenhead,
                                      int64_t skip,
                                      int64_t start,
                                      int64_t step,
                                      uint64_t invocation_index,
                                      uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) / lenhead;
    int64_t thready_id = (blockIdx.x * blockDim.x + threadIdx.x) % lenhead;

    if (thread_id < lencarry) {
      nextcarryptr[(thread_id * lenhead) + thready_id] =
          ((skip * carryptr[thread_id]) + start) + (thready_id * step);
    }
  }
}
