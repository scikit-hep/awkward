// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class INDEXEDARRAY_VALIDITY_ERRORS {
  IND_LT_0,   // message: "index[i] < 0"
  IND_GT_LEN  // message: "index[i] >= len(content)"
};

template <typename T>
__global__ void
awkward_IndexedArray_validity(
    const T* index,
    int64_t length,
    int64_t lencontent,
    bool isoption,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      T idx = index[thread_id];
      if (!isoption) {
        if (idx < 0) {
          RAISE_ERROR(INDEXEDARRAY_VALIDITY_ERRORS::IND_LT_0)
        }
      }
      if (idx >= lencontent) {
        RAISE_ERROR(INDEXEDARRAY_VALIDITY_ERRORS::IND_GT_LEN)
      }
    }
  }
}
