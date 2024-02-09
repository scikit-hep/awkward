// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class LISTOFFSETARRAY_TOREGULARARRAY_ERRORS {
  OFF_DEC,     // message: "offsets must be monotonically increasing"
  LEN_NOT_REG, // message: "cannot convert to RegularArray because subarray lengths are not regular"
};

template <typename T, typename C>
__global__ void
awkward_ListOffsetArray_toRegularArray(
    T* size,
    const C* fromoffsets,
    int64_t offsetslength,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    *size = offsetslength > 1 ? (int64_t)fromoffsets[1] - (int64_t)fromoffsets[0] : -1;

    if (thread_id < offsetslength - 1) {
      int64_t count = (int64_t)fromoffsets[thread_id + 1] - (int64_t)fromoffsets[thread_id];
      if (count < 0) {
        RAISE_ERROR(LISTOFFSETARRAY_TOREGULARARRAY_ERRORS::OFF_DEC)
      }
      else if (*size != count) {
        RAISE_ERROR(LISTOFFSETARRAY_TOREGULARARRAY_ERRORS::LEN_NOT_REG)
      }
    }
    if (*size == -1) {
      *size = 0;
    }
  }
}
