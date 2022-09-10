// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, length, stride, invocation_index, err_code) = args
//     scan_in_array = cupy.empty(length, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_NumpyArray_getitem_boolean_nonzero_a", toptr.dtype, fromptr.dtype]))(grid, block, (toptr, fromptr, length, stride, scan_in_array, invocation_index, err_code))
//     scan_in_array = inclusive_scan(grid, block, (scan_in_array, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_NumpyArray_getitem_boolean_nonzero_b", toptr.dtype, fromptr.dtype]))(grid, block, (toptr, fromptr, length, stride, scan_in_array, invocation_index, err_code))
// out["awkward_NumpyArray_getitem_boolean_nonzero_a", {dtype_specializations}] = None
// out["awkward_NumpyArray_getitem_boolean_nonzero_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C>
__global__ void
awkward_NumpyArray_getitem_boolean_nonzero_a(T* toptr,
                                             const C* fromptr,
                                             int64_t length,
                                             int64_t stride,
                                             int64_t* scan_in_array,
                                             uint64_t invocation_index,
                                             uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      if (thread_id % stride == 0) {
        if (fromptr[thread_id] != 0) {
          scan_in_array[thread_id] = 1;
        }
      } else {
        scan_in_array[thread_id] = 0;
      }
    }
  }
}

template <typename T, typename C>
__global__ void
awkward_NumpyArray_getitem_boolean_nonzero_b(T* toptr,
                                             const C* fromptr,
                                             int64_t length,
                                             int64_t stride,
                                             int64_t* scan_in_array,
                                             uint64_t invocation_index,
                                             uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length && thread_id % stride == 0) {
      if (fromptr[thread_id] != 0) {
        toptr[scan_in_array[thread_id] - 1] = thread_id;
      }
    }
  }
}
