// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C>
__global__ void
awkward_RegularArray_getitem_carry(
    T* tocarry,
    const C* fromcarry,
    int64_t lencarry,
    int64_t size,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) / size;
    int64_t thready_id = (blockIdx.x * blockDim.x + threadIdx.x) % size;

    if (thread_id < lencarry) {
      tocarry[(thread_id * size) + thready_id] =
          (fromcarry[thread_id] * size) + thready_id;
    }
  }
}
