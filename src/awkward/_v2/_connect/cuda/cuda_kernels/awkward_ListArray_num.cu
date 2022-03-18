// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C>
__global__ void
cuda_ListArray_num(C* tonum, const T* fromstarts, const T* fromstops, int64_t length, int64_t invocation_index, int64_t* err_code) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id < length && err_code[0] == MAX_NUMPY_INT) {
    int64_t start = fromstarts[thread_id];
    int64_t stop = fromstops[thread_id];
    tonum[thread_id] = (C)(stop - start);
  }
}
