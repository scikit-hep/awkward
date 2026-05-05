// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C, typename U>
__global__ void
awkward_RegularArray_getitem_jagged_expand(
    T* multistarts,
    C* multistops,
    const U* singleoffsets,
    int64_t regularsize,
    int64_t regularlength,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) / regularsize;
    int64_t thready_id = (blockIdx.x * blockDim.x + threadIdx.x) % regularsize;

    if (thread_id < regularlength) {
      multistarts[(thread_id * regularsize) + thready_id] =
          singleoffsets[thready_id];
      multistops[(thread_id * regularsize) + thready_id] =
          singleoffsets[thready_id + 1];
    }
  }
}
