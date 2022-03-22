// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__global__ void
awkward_RegularArray_getitem_next_at(int64_t* tocarry,
                                     int64_t at,
                                     int64_t length,
                                     int64_t size,
                                     uint64_t invocation_index,
                                     uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    auto regular_at = at;
    if ((regular_at < 0)) {
      regular_at += size;

    } else {
    }
    if (!(0 <= regular_at) && (regular_at < size)) {
      err->str = "index out of range";
      err->filename = FILENAME(__LINE__);
      err->pass_through = true;

    } else {
    }
    if (thread_id < length) {
      tocarry[thread_id] = ((thread_id * size) + regular_at);
    }
  }
}
