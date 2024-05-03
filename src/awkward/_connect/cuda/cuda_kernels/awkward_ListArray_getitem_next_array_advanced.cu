// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class LISTARRAY_GETITEM_NEXT_ARRAY_ADVANCED_ERRORS {
  STOP_LT_START,     // message: "stops[i] < starts[i]"
  STOP_GET_LEN,      // message: "stops[i] > len(content)"
  IND_OUT_OF_RANGE,  // message: "index out of range"
};

template <typename T, typename C, typename U, typename V, typename W, typename X>
__global__ void
awkward_ListArray_getitem_next_array_advanced(
    T* tocarry,
    C* toadvanced,
    const U* fromstarts,
    const V* fromstops,
    const W* fromarray,
    const X* fromadvanced,
    int64_t lenstarts,
    int64_t lencontent,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < lenstarts) {
      if (fromstops[thread_id] < fromstarts[thread_id]) {
        RAISE_ERROR(LISTARRAY_GETITEM_NEXT_ARRAY_ADVANCED_ERRORS::STOP_LT_START)
      }
      if ((fromstarts[thread_id] != fromstops[thread_id]) &&
          (fromstops[thread_id] > lencontent)) {
        RAISE_ERROR(LISTARRAY_GETITEM_NEXT_ARRAY_ADVANCED_ERRORS::STOP_GET_LEN)
      }
      int64_t length = fromstops[thread_id] - fromstarts[thread_id];
      int64_t regular_at = fromarray[fromadvanced[thread_id]];
      if (regular_at < 0) {
        regular_at += length;
      }
      if (!(0 <= regular_at && regular_at < length)) {
        RAISE_ERROR(LISTARRAY_GETITEM_NEXT_ARRAY_ADVANCED_ERRORS::IND_OUT_OF_RANGE)
      }
      tocarry[thread_id] = fromstarts[thread_id] + regular_at;
      toadvanced[thread_id] = thread_id;
    }
  }
}
