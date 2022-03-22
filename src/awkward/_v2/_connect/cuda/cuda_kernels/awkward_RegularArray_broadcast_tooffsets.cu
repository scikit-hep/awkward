// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__global__ void
awkward_RegularArray_broadcast_tooffsets(const int64_t* fromoffsets,
                                         int64_t offsetslength,
                                         int64_t size,
                                         uint64_t invocation_index,
                                         uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < (offsetslength - 1)) {
      auto count =
          (int)((fromoffsets[(thread_id + 1)] - fromoffsets[thread_id]));
      if ((count < 0)) {
        err->str = "broadcast's offsets must be monotonically increasing";
        err->filename = FILENAME(__LINE__);
        err->pass_through = true;

      } else {
      }
      if ((size != count)) {
        err->str = "cannot broadcast nested list";
        err->filename = FILENAME(__LINE__);
        err->pass_through = true;

      } else {
      }
    }
  }
}
