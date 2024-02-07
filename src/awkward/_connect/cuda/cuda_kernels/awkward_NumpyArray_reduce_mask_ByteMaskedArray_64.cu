// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C>
__global__ void
awkward_NumpyArray_reduce_mask_ByteMaskedArray_64(
    T* toptr,
    const C* parents,
    int64_t lenparents,
    int64_t outlength,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < outlength) {
      toptr[thread_id] = 1;
    }
    if (thread_id < lenparents) {
      toptr[parents[thread_id]] = 0;
    }
  }
}
