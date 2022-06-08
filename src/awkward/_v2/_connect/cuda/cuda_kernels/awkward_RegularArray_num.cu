// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T>
__global__ void
awkward_RegularArray_num(T* tonum,
                         int64_t size,
                         int64_t length,
                         uint64_t invocation_index,
                         uint64_t* err_code) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (err_code[0] == NO_ERROR) {
    if (thread_id < length) {
      tonum[thread_id] = size;
    }
  }
}
