// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C, typename U>
__global__ void
awkward_ListArray_num(T* tonum,
                      const C* fromstarts,
                      const U* fromstops,
                      int64_t length,
                      uint64_t invocation_index,
                      uint64_t* err_code) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (err_code[0] == NO_ERROR) {
    if (thread_id < length) {
      int64_t start = fromstarts[thread_id];
      int64_t stop = fromstops[thread_id];
      tonum[thread_id] = (C)(stop - start);
    }
  }
}
