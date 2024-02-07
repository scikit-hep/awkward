// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class LISTARRAY_GETITEM_JAGGED_EXPAND_ERRORS {
  STOPS_LT_START,  // message: "stops[i] < starts[i]"
  FIT_ERR          // message: "cannot fit jagged slice into nested list"
};

template <typename T, typename C, typename U, typename V, typename W, typename X>
__global__ void
awkward_ListArray_getitem_jagged_expand(
    T* multistarts,
    C* multistops,
    const U* singleoffsets,
    V* tocarry,
    const W* fromstarts,
    const X* fromstops,
    int64_t jaggedsize,
    int64_t length,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) / jaggedsize;
    int64_t thready_id = (blockIdx.x * blockDim.x + threadIdx.x) % jaggedsize;
    if (thread_id < length && thready_id < jaggedsize) {
      W start = fromstarts[thread_id];
      X stop = fromstops[thread_id];
      if (stop < start) {
        RAISE_ERROR(LISTARRAY_GETITEM_JAGGED_EXPAND_ERRORS::STOPS_LT_START)
      }
      if ((stop - start) != jaggedsize) {
        RAISE_ERROR(LISTARRAY_GETITEM_JAGGED_EXPAND_ERRORS::FIT_ERR)
      }
      multistarts[(thread_id * jaggedsize) + thready_id] =
          singleoffsets[thready_id];
      multistops[(thread_id * jaggedsize) + thready_id] =
          singleoffsets[(thready_id + 1)];
      tocarry[(thread_id * jaggedsize) + thready_id] = (start + thready_id);
    }
  }
}
