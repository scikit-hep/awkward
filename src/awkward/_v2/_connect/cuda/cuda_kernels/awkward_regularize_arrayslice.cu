// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__global__ void
awkward_regularize_arrayslice(int64_t* flatheadptr,
                              int64_t lenflathead,
                              int64_t length,
                              uint64_t invocation_index,
                              uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < lenflathead) {
      auto original = flatheadptr[thread_id];
      if ((flatheadptr[thread_id] < 0)) {
        flatheadptr[thread_id] += length;

      } else {
      }
      if ((flatheadptr[thread_id] < 0) || (flatheadptr[thread_id] >= length)) {
        err->str = "index out of range";
        err->filename = FILENAME(__LINE__);
        err->pass_through = true;

      } else {
      }
    }
  }
}
