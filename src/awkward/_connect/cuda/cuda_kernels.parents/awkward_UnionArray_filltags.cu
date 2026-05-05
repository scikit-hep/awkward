// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C>
__global__ void
awkward_UnionArray_filltags(
    T* totags,
    int64_t totagsoffset,
    const C* fromtags,
    int64_t length,
    int64_t base,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      totags[totagsoffset + thread_id] = (T)(fromtags[thread_id] + base);
    }
  }
}
