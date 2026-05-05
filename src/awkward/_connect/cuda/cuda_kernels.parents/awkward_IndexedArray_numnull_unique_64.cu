// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

template <typename T>
__global__ void
awkward_IndexedArray_numnull_unique_64(
    T* toindex,
    int64_t lenindex,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id <= lenindex) {
      toindex[thread_id] = (thread_id < lenindex ? thread_id : -1);
    }
  }
}
