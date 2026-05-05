// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C>
__global__ void
awkward_IndexedArray_reduce_next_fix_offsets_64(
    T* outoffsets,
    const C* starts,
    int64_t startslength,
    int64_t outindexlength,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < startslength) {
      outoffsets[thread_id] = starts[thread_id];
    }
    outoffsets[startslength] = outindexlength;
  }
}
