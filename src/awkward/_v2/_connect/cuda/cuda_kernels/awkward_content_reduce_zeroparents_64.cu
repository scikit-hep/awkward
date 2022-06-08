// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T>
__global__ void
awkward_content_reduce_zeroparents_64(T* toparents,
                                      int64_t length,
                                      uint64_t invocation_index,
                                      uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      toparents[thread_id] = 0;
    }
  }
}
