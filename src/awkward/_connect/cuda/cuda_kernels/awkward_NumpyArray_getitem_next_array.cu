// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_NumpyArray_getitem_next_array(T* nextcarryptr,
                                      C* nextadvancedptr,
                                      const U* carryptr,
                                      const V* flatheadptr,
                                      int64_t lencarry,
                                      int64_t lenflathead,
                                      int64_t skip,
                                      uint64_t invocation_index,
                                      uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) / lenflathead;
    int64_t thready_id = (blockIdx.x * blockDim.x + threadIdx.x) % lenflathead;

    if (thread_id < lencarry) {
      nextcarryptr[(thread_id * lenflathead) + thready_id] =
          (skip * carryptr[thread_id]) + flatheadptr[thready_id];
      nextadvancedptr[(thread_id * lenflathead) + thready_id] = thready_id;
    }
  }
}
