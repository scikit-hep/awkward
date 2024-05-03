// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C>
__global__ void
awkward_missing_repeat(
    T* outindex,
    const C* index,
    int64_t indexlength,
    int64_t repetitions,
    int64_t regularsize,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) / indexlength;
    int64_t thready_id = (blockIdx.x * blockDim.x + threadIdx.x) % indexlength;
    if (thread_id < repetitions) {
      T base = index[thready_id];
      outindex[thread_id * indexlength + thready_id] =
          base + (base >= 0 ? thread_id * regularsize : 0);
    }
  }
}
