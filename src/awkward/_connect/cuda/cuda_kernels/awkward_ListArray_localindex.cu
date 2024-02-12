// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C>
__global__ void
awkward_ListArray_localindex(
    T* toindex,
    const C* offsets,
    int64_t length,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      int64_t start = (int64_t)offsets[thread_id];
      int64_t stop = (int64_t)offsets[thread_id + 1];
      for (int64_t j = start;  j < stop;  j++) {
        toindex[j] = j - start;
      }
    }
  }
}
