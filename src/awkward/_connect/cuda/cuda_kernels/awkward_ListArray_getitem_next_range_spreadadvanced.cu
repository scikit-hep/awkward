// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C, typename U>
__global__ void
awkward_ListArray_getitem_next_range_spreadadvanced(
    T* toadvanced,
    const C* fromadvanced,
    const U* fromoffsets,
    int64_t lenstarts,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < lenstarts) {
      C count = fromoffsets[thread_id + 1] - fromoffsets[thread_id];
      for (int64_t j = 0;  j < count;  j++) {
        toadvanced[fromoffsets[thread_id] + j] = fromadvanced[thread_id];
      }
    }
  }
}
