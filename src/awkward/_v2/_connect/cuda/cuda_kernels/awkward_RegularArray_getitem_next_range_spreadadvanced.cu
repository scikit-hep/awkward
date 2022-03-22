// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__global__ void
awkward_RegularArray_getitem_next_range_spreadadvanced(
    int64_t* toadvanced,
    const int64_t* fromadvanced,
    int64_t length,
    int64_t nextsize,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      if (thready_dim < nextsize) {
        toadvanced[((thread_id * nextsize) + thready_dim)] =
            fromadvanced[thread_id];
      }
    }
  }
}
