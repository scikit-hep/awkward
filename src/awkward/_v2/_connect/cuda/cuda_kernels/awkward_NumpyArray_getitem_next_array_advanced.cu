// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C, typename U, typename W>
__global__ void
awkward_NumpyArray_getitem_next_array_advanced(T* nextcarryptr,
                                               const C* carryptr,
                                               const U* advancedptr,
                                               const W* flatheadptr,
                                               int64_t lencarry,
                                               int64_t skip,
                                               uint64_t invocation_index,
                                               uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < lencarry) {
      nextcarryptr[thread_id] =
          (skip * carryptr[thread_id]) + flatheadptr[advancedptr[thread_id]];
    }
  }
}
