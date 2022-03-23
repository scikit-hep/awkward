// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T>
__global__ void
awkward_RegularArray_compact_offsets(T* tooffsets,
                                     int64_t length,
                                     int64_t size,
                                     uint64_t invocation_index,
                                     uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    tooffsets[0] = 0;
    if (thread_id < length) {
      tooffsets[thread_id + 1] = (thread_id + 1) * size;
    }
  }
}
