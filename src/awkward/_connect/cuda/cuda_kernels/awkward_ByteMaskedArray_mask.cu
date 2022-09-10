// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C>
__global__ void
awkward_ByteMaskedArray_mask(T* tomask,
                             const C* frommask,
                             int64_t length,
                             bool validwhen,
                             uint64_t invocation_index,
                             uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      tomask[thread_id] = (frommask[thread_id] != 0) != validwhen;
    }
  }
}
