// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class IDENTITIES_GETITEM_CARRY {
  IND_OUT_OF_RANGE,  // message: "index out of range"
};

template <typename T, typename C, typename U>
__global__ void
awkward_Identities_getitem_carry(T* newidentitiesptr,
                                 const C* identitiesptr,
                                 const U* carryptr,
                                 int64_t lencarry,
                                 int64_t width,
                                 int64_t length,
                                 uint64_t invocation_index,
                                 uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) / width;
    int64_t thready_id = (blockIdx.x * blockDim.x + threadIdx.x) % width;
    if (thread_id < lencarry) {
      if (carryptr[thread_id] >= length) {
        RAISE_ERROR(IDENTITIES_GETITEM_CARRY::IND_OUT_OF_RANGE)
      }
      newidentitiesptr[width * thread_id + thready_id] =
          identitiesptr[width * carryptr[thread_id] + thready_id];
    }
  }
}
