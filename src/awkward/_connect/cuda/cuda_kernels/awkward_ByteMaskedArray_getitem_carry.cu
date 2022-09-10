// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class BYTEMASKEDARRAY_GETITEM_CARRY_ERRORS {
  IND_OUT_OF_RANGE  //  message: "index out of range"
};

template <typename T, typename C, typename U>
__global__ void
awkward_ByteMaskedArray_getitem_carry(T* tomask,
                                      const C* frommask,
                                      int64_t lenmask,
                                      const U* fromcarry,
                                      int64_t lencarry,
                                      uint64_t invocation_index,
                                      uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < lencarry) {
      if (fromcarry[thread_id] >= lenmask) {
        RAISE_ERROR(BYTEMASKEDARRAY_GETITEM_CARRY_ERRORS::IND_OUT_OF_RANGE)
      }
      tomask[thread_id] = frommask[fromcarry[thread_id]];
    }
  }
}
