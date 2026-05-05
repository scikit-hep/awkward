// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C, typename U, typename V, typename W, typename X>
__global__ void
awkward_UnionArray_simplify(
    T* totags,
    C* toindex,
    const U* outertags,
    const V* outerindex,
    const W* innertags,
    const X* innerindex,
    int64_t towhich,
    int64_t innerwhich,
    int64_t outerwhich,
    int64_t length,
    int64_t base,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      if (outertags[thread_id] == outerwhich) {
        V j = outerindex[thread_id];
        if (innertags[j] == innerwhich) {
          totags[thread_id] = (T)towhich;
          toindex[thread_id] = (C)(innerindex[j] + base);
        }
      }
    }
  }
}
