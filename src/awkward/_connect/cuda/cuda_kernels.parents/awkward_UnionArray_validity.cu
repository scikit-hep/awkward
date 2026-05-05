// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class UNIONARRAY_VALIDITY_ERRORS {
  TAGS_LT_0,    // message: "tags[i] < 0"
  INDEX_LT_0,   // message: "index[i] < 0"
  TAGS_GT_LEN,  // message: "tags[i] >= len(contents)"
  IND_GT_LEN    // message: "index[i] >= len(content[tags[i]])"
};

template <typename T, typename C, typename U>
__global__ void
awkward_UnionArray_validity(
    const T* tags,
    const C* index,
    int64_t length,
    int64_t numcontents,
    const U* lencontents,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      T tag = tags[thread_id];
      C idx = index[thread_id];
      if (tag < 0) {
        RAISE_ERROR(UNIONARRAY_VALIDITY_ERRORS::TAGS_LT_0)
      }
      if (idx < 0) {
        RAISE_ERROR(UNIONARRAY_VALIDITY_ERRORS::INDEX_LT_0)
      }
      if (tag >= numcontents) {
        RAISE_ERROR(UNIONARRAY_VALIDITY_ERRORS::TAGS_GT_LEN)
      }
      int64_t lencontent = lencontents[tag];
      if ((idx >= lencontent)) {
        RAISE_ERROR(UNIONARRAY_VALIDITY_ERRORS::IND_GT_LEN)
      }
    }
  }
}
