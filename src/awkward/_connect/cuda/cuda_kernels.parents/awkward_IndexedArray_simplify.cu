// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class INDEXEDARRAY_SIMPLIFY_ERRORS {
  IND_OUT_OF_RANGE  // message: "index out of range"
};

template <typename T, typename C, typename U>
__global__ void
awkward_IndexedArray_simplify(
    T* toindex,
    const C* outerindex,
    int64_t outerlength,
    const U* innerindex,
    int64_t innerlength,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < outerlength) {
      C j = outerindex[thread_id];
      if (j < 0) {
        toindex[thread_id] = -1;

      } else {
        if (j >= innerlength) {
          RAISE_ERROR(INDEXEDARRAY_SIMPLIFY_ERRORS::IND_OUT_OF_RANGE)
        } else {
          toindex[thread_id] = innerindex[j];
        }
      }
    }
  }
}
