// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C>
__global__ void
awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64(
    T* nextstarts,
    const C* nextparents,
    int64_t nextlen,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t lastnextparent = -1;

    if (thread_id < nextlen) {
      if (thread_id != 0) {
        lastnextparent = nextparents[thread_id - 1];
      }
      if (nextparents[thread_id] != lastnextparent) {
        nextstarts[nextparents[thread_id]] = thread_id;
      }
    }
  }
}
