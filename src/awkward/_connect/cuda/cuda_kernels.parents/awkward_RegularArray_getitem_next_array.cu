// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C, typename U>
__global__ void
awkward_RegularArray_getitem_next_array(
    T* tocarry,
    C* toadvanced,
    const U* fromarray,
    int64_t length,
    int64_t lenarray,
    int64_t size,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) / lenarray;
    int64_t thready_id = (blockIdx.x * blockDim.x + threadIdx.x) % lenarray;

    if (thread_id < length) {
      tocarry[(thread_id * lenarray) + thready_id] =
          (thread_id * size) + fromarray[thready_id];
      toadvanced[(thread_id * lenarray) + thready_id] = thready_id;
    }
  }
}
