// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__global__ void
awkward_RegularArray_getitem_jagged_expand(int64_t* multistarts,
                                           int64_t* multistops,
                                           const int64_t* singleoffsets,
                                           int64_t regularsize,
                                           int64_t regularlength,
                                           uint64_t invocation_index,
                                           uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < regularlength) {
      if (thready_dim < regularsize) {
        multistarts[((thread_id * regularsize) + thready_dim)] =
            singleoffsets[thready_dim];
        multistops[((thread_id * regularsize) + thready_dim)] =
            singleoffsets[(thready_dim + 1)];
      }
    }
  }
}
