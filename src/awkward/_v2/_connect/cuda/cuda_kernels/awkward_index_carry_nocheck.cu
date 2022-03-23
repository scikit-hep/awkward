// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C, typename U>
__global__ void
awkward_index_carry_nocheck(T* toindex,
                            const C* fromindex,
                            const U* carry,
                            int64_t length,
                            uint64_t invocation_index,
                            uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      toindex[thread_id] = fromindex[(int64_t)carry[thread_id]];
    }
  }
}
