// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_NumpyArray_reduce_adjust_starts_shifts_64(
    T* toptr,
    int64_t outlength,
    const C* parents,
    const U* starts,
    const V* shifts,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < outlength) {
      int64_t i = toptr[thread_id];
      if (i >= 0) {
        int64_t parent = parents[i];
        int64_t start = starts[parent];
        toptr[thread_id] += (shifts[i] - start);
      }
    }
  }
}
