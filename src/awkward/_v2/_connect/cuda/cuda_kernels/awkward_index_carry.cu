// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class INDEX_CARRY_ERRORS {
  IND_OUT_OF_RANGE  // message: "index out of range"
};

template <typename T, typename C, typename U>
__global__ void
awkward_index_carry(T* toindex,
                    const C* fromindex,
                    const U* carry,
                    int64_t lenfromindex,
                    int64_t length,
                    uint64_t invocation_index,
                    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      U j = carry[thread_id];
      if (j > lenfromindex) {
        RAISE_ERROR(INDEX_CARRY_ERRORS::IND_OUT_OF_RANGE)
      }
      toindex[thread_id] = fromindex[(int64_t)j];
    }
  }
}
