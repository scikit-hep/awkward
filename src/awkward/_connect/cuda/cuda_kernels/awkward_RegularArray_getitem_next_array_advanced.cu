// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_RegularArray_getitem_next_array_advanced(
    T* tocarry,
    C* toadvanced,
    const U* fromadvanced,
    const V* fromarray,
    int64_t length,
    int64_t size,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      tocarry[thread_id] =
          (thread_id * size) + fromarray[fromadvanced[thread_id]];
      toadvanced[thread_id] = thread_id;
    }
  }
}
