// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// In the offsets-pipeline migration, `parents[toptr[k]] == k` by construction
// of argmin/argmax (the result for output bin k is by definition an element
// of bin k), so the parent indirection collapses to using the thread/bin
// index directly. The `offsets` slot is preserved in the signature for
// symmetry with the rest of the pipeline; downstream callers may drop it.
template <typename T, typename C, typename U>
__global__ void
awkward_NumpyArray_reduce_adjust_starts_64(
    T* toptr,
    int64_t outlength,
    const C* /* offsets */,
    const U* starts,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < outlength) {
      T i = toptr[thread_id];
      if (i >= 0) {
        toptr[thread_id] -= starts[thread_id];
      }
    }
  }
}
