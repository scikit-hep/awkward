// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename FROM, typename TO>
__global__ void
awkward_IndexedArray_fill(
    TO* toindex,
    int64_t toindexoffset,
    const FROM* fromindex,
    int64_t length,
    int64_t base,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      FROM fromval = fromindex[thread_id];
      toindex[toindexoffset + thread_id] = fromval < 0 ? (TO)-1 : (TO)(fromval + base);
    }
  }
}
