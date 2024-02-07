// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C, typename U>
__global__ void
awkward_ByteMaskedArray_overlay_mask(
    T* tomask,
    const C* theirmask,
    const U* mymask,
    int64_t length,
    bool validwhen,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      bool theirs = theirmask[thread_id];
      bool mine = (mymask[thread_id] != 0) != validwhen;
      tomask[thread_id] = ((theirs | mine) ? 1 : 0);
    }
  }
}
