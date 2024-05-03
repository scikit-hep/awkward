// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_UnionArray_simplify_one(
    T* totags,
    C* toindex,
    const U* fromtags,
    const V* fromindex,
    int64_t towhich,
    int64_t fromwhich,
    int64_t length,
    int64_t base,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      if (fromtags[thread_id] == fromwhich) {
        totags[thread_id] = (T)towhich;
        toindex[thread_id] = (C)(fromindex[thread_id] + base);
      }
    }
  }
}
