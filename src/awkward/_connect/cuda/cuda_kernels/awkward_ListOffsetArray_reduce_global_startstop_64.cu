// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C, typename U>
__global__ void
awkward_ListOffsetArray_reduce_global_startstop_64(T* globalstart,
                                                   C* globalstop,
                                                   const U* offsets,
                                                   int64_t length,
                                                   uint64_t invocation_index,
                                                   uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    globalstart[0] = offsets[0];
    globalstop[0] = offsets[length];
  }
}
