// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C, typename U>
__global__ void
awkward_ListOffsetArray_flatten_offsets(
    T* tooffsets,
    const C* outeroffsets,
    int64_t outeroffsetslen,
    const U* inneroffsets,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < outeroffsetslen) {
      tooffsets[thread_id] = inneroffsets[outeroffsets[thread_id]];
    }
  }
}
