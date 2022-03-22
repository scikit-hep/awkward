// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C>
__global__ void
awkward_NumpyArray_getitem_next_at(T* nextcarryptr,
                                   const C* carryptr,
                                   int64_t lencarry,
                                   int64_t skip,
                                   int64_t at,
                                   uint64_t invocation_index,
                                   uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < lencarry) {
      nextcarryptr[thread_id] = (skip * carryptr[thread_id]) + at;
    }
  }
}
