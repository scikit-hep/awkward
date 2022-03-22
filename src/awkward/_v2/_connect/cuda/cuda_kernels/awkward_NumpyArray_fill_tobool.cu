// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C>
__global__ void
awkward_NumpyArray_fill_tobool(T* toptr,
                               int64_t tooffset,
                               const C* fromptr,
                               int64_t length,
                               uint64_t invocation_index,
                               uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      toptr[tooffset + thread_id] = fromptr[thread_id] > 0 ? true : false;
    }
  }
}
