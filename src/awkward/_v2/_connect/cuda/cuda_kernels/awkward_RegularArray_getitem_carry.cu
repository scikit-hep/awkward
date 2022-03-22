// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__global__ void
awkward_RegularArray_getitem_carry(int64_t* tocarry,
                                   const int64_t* fromcarry,
                                   int64_t lencarry,
                                   int64_t size,
                                   uint64_t invocation_index,
                                   uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < lencarry) {
      if (thready_dim < size) {
        tocarry[((thread_id * size) + thready_dim)] =
            ((fromcarry[thread_id] * size) + thready_dim);
      }
    }
  }
}
