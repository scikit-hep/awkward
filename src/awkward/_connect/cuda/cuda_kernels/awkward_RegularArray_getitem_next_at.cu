// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


// BEGIN PYTHON
// def f(grid, block, args):
//     (tocarry, at, length, size, invocation_index, err_code) = args
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_RegularArray_getitem_next_at", tocarry.dtype]))(grid, block, (tocarry, cupy.array(at), length, size, invocation_index, err_code))
// out["awkward_RegularArray_getitem_next_at", {dtype_specializations}] = None
// END PYTHON

enum class REGULARARRAY_GETITEM_NEXT_AT_ERRORS {
  IND_OUT_OF_RANGE  // message: "index out of range"
};

template <typename T>
__global__ void
awkward_RegularArray_getitem_next_at(
    T* tocarry,
    int64_t* at,
    int64_t length,
    int64_t size,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t regular_at = at[0];
    if (regular_at < 0) {
      regular_at += size;
    }
    if (!(0 <= regular_at && regular_at < size)) {
      RAISE_ERROR(REGULARARRAY_GETITEM_NEXT_AT_ERRORS::IND_OUT_OF_RANGE)
    }
    if (thread_id < length) {
      tocarry[thread_id] = (thread_id * size) + regular_at;
    }
  }
}
