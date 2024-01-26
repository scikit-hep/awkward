// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C, typename U>
__global__ void
awkward_ListArray_min_range(T* tomin,
                            const C* fromstarts,
                            const U* fromstops,
                            int64_t lenstarts,
                            uint64_t invocation_index,
                            uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t shorter = fromstops[0] - fromstarts[0];

    if (thread_id >=1 && thread_id < lenstarts) {
      int64_t rangeval = fromstops[thread_id] - fromstarts[thread_id];
      shorter = (shorter < rangeval) ? shorter : rangeval;
      atomicMin(tomin, shorter);
    }
  }
}
