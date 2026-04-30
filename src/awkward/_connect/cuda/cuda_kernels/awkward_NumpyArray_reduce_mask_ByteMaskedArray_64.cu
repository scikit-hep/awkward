// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

// Note: When invoking this kernel, the grid size should be based on outlength rather than lenparents:
// int64_t blocks = (outlength + threads_per_block - 1) / threads_per_block;
//

template <typename T, typename C>
__global__ void
awkward_NumpyArray_reduce_mask_ByteMaskedArray_offsets_64(
    T* toptr,
    const C* offsets,
    int64_t outlength,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < outlength) {
      // The mask value (toptr) is 1 if the bin is empty, 0 if it has content.
      // Bin i's content is defined by the range [offsets[thread_id], offsets[thread_id + 1]).
      if (offsets[thread_id + 1] - offsets[thread_id] > 0) {
        toptr[thread_id] = 0;
      } else {
        toptr[thread_id] = 1;
      }
    }
  }
}
