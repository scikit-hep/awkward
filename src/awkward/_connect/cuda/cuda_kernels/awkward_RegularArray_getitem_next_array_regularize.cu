// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class REGULARARRAY_GETITEM_NEXT_ARRAY_REGULARIZE_ERRORS {
  IND_OUT_OF_RANGE  // message: "index out of range"
};

template <typename T, typename C>
__global__ void
awkward_RegularArray_getitem_next_array_regularize(
    T* toarray,
    const C* fromarray,
    int64_t lenarray,
    int64_t size,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < lenarray) {
      toarray[thread_id] = fromarray[thread_id];
      if (toarray[thread_id] < 0) {
        toarray[thread_id] = fromarray[thread_id] + size;
      }
      if (!(0 <= toarray[thread_id]  &&  toarray[thread_id] < size)) {
        RAISE_ERROR(REGULARARRAY_GETITEM_NEXT_ARRAY_REGULARIZE_ERRORS::IND_OUT_OF_RANGE)
      }
    }
  }
}
