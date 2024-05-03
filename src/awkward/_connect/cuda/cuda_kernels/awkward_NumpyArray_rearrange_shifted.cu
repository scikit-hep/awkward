// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromshifts, length, fromoffsets, offsetslength, fromparents, fromstarts, invocation_index, err_code) = args
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_NumpyArray_rearrange_shifted_a", toptr.dtype, fromshifts.dtype, fromoffsets.dtype, fromparents.dtype, fromstarts.dtype]))(grid, block, (toptr, fromshifts, length, fromoffsets, offsetslength, fromparents, fromstarts, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_NumpyArray_rearrange_shifted_b", toptr.dtype, fromshifts.dtype, fromoffsets.dtype, fromparents.dtype, fromstarts.dtype]))(grid, block, (toptr, fromshifts, length, fromoffsets, offsetslength, fromparents, fromstarts, invocation_index, err_code))
// out["awkward_NumpyArray_rearrange_shifted_a", {dtype_specializations}] = None
// out["awkward_NumpyArray_rearrange_shifted_b", {dtype_specializations}] = None
// END PYTHON


template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_NumpyArray_rearrange_shifted_a(
    T* toptr,
    C* fromshifts,
    int64_t length,
    U* fromoffsets,
    int64_t offsetslength,
    V* fromparents,
    W* fromstarts,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < offsetslength - 1) {
      for (int64_t j = 0; j < fromoffsets[thread_id + 1] - fromoffsets[thread_id]; j++) {
        int64_t idx = fromoffsets[thread_id] + j;
        toptr[idx] = toptr[idx] + fromoffsets[thread_id];
      }
    }
  }
}

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_NumpyArray_rearrange_shifted_b(
    T* toptr,
    C* fromshifts,
    int64_t length,
    U* fromoffsets,
    int64_t offsetslength,
    V* fromparents,
    W* fromstarts,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      int64_t parent = fromparents[thread_id];
      int64_t start = fromstarts[parent];
      toptr[thread_id] = toptr[thread_id] + fromshifts[toptr[thread_id]] - start;
    }
  }
}
