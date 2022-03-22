// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__global__ void
awkward_NumpyArray_reduce_adjust_starts_64(int64_t* toptr,
                                           int64_t outlength,
                                           const int64_t* parents,
                                           const int64_t* starts,
                                           uint64_t invocation_index,
                                           uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < outlength) {
      auto i = toptr[thread_id];
      if ((i >= 0)) {
        auto parent = parents[i];

        auto start = starts[parent];

        toptr[thread_id] += -start;

      } else {
      }
    }
  }
}
