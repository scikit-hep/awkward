// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C>
__global__ void
awkward_NumpyArray_contiguous_next(T* topos,
                                   const C* frompos,
                                   int64_t length,
                                   int64_t skip,
                                   int64_t stride,
                                   uint64_t invocation_index,
                                   uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) / skip;
    int64_t thready_id = (blockIdx.x * blockDim.x + threadIdx.x) % skip;

    if (thread_id < length) {
      topos[(thread_id * skip) + thready_id] =
          frompos[thread_id] + (thready_id * stride);
    }
  }
}
