// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_ListArray_fill(
    T* tostarts,
    int64_t tostartsoffset,
    C* tostops,
    int64_t tostopsoffset,
    const U* fromstarts,
    const V* fromstops,
    int64_t length,
    int64_t base,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      tostarts[tostartsoffset + thread_id] = (T)(fromstarts[thread_id] + base);
      tostops[tostopsoffset + thread_id] = (C)(fromstops[thread_id] + base);
    }
  }
}
