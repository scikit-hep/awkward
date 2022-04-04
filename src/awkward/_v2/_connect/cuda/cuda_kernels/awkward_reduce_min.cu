template <typename T, typename C, typename U>
__global__ void
awkward_reduce_min(T* toptr,
                   const C* fromptr,
                   const U* parents,
                   int64_t lenparents,
                   int64_t outlength,
                   T identity,
                   uint64_t invocation_index,
                   uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < outlength) {
      toptr[thread_id] = identity;
      toptr[parents[thread_id]] = identity;
    }
    if (thread_id < lenparents) {
      C x = fromptr[thread_id];
      toptr[parents[thread_id]] =
          (x < toptr[parents[thread_id]] ? x : toptr[parents[thread_id]]);
    }
  }
}
