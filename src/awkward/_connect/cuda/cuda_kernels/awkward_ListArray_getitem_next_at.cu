// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class LISTARRAY_GETITEM_NEXT_AT_ERRORS {
  IND_OUT_OF_RANGE,  // message: "index out of range"
};

template <typename T, typename C, typename U>
__global__ void
awkward_ListArray_getitem_next_at(
    T* tocarry,
    const C* fromstarts,
    const U* fromstops,
    int64_t lenstarts,
    int64_t at,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < lenstarts) {
      int64_t length = fromstops[thread_id] - fromstarts[thread_id];
      int64_t regular_at = at;
      if (regular_at < 0) {
        regular_at += length;
      }
      if (!(0 <= regular_at && regular_at < length)) {
        RAISE_ERROR(LISTARRAY_GETITEM_NEXT_AT_ERRORS::IND_OUT_OF_RANGE)
      }
      tocarry[thread_id] = fromstarts[thread_id] + regular_at;
    }
  }
}
