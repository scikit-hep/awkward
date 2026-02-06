// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T_toindex, typename T_mask, typename T_fromindex>
__global__ void
awkward_IndexedArray_overlay_mask(
    T_toindex* toindex,
    const T_mask* mask,
    const T_fromindex* fromindex,
    int64_t length,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      T_mask m = mask[thread_id];
      toindex[thread_id] = m ? (T_toindex)(-1) : (T_toindex)fromindex[thread_id];
    }
  }
}
