// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, parents, lenparents, outlength, invocation_index, err_code) = args
//     scan_in_array = cupy.array(toptr, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_sum_a", toptr.dtype, fromptr.dtype, parents.dtype]))(grid, block, (toptr, fromptr, parents, lenparents, outlength, scan_in_array, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_sum_b", toptr.dtype, fromptr.dtype, parents.dtype]))(grid, block, (toptr, fromptr, parents, lenparents, outlength, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_sum_c", toptr.dtype, fromptr.dtype, parents.dtype]))(grid, block, (toptr, fromptr, parents, lenparents, outlength, scan_in_array, invocation_index, err_code))
// out["awkward_reduce_sum_a", {dtype_specializations}] = None
// out["awkward_reduce_sum_b", {dtype_specializations}] = None
// out["awkward_reduce_sum_c", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_sum_a(T* toptr,
                     const C* fromptr,
                     const U* parents,
                     int64_t lenparents,
                     int64_t outlength,
                     int64_t* scan_in_array,
                     uint64_t invocation_index,
                     uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < outlength) {
      scan_in_array[thread_id] = 0;
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_sum_b(T* toptr,
                     const C* fromptr,
                     const U* parents,
                     int64_t lenparents,
                     int64_t outlength,
                     int64_t* scan_in_array,
                     uint64_t invocation_index,
                     uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < lenparents) {
      scan_in_array[thread_id] = (T)(parents[thread_id] + fromptr[thread_id]);
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_sum_c(T* toptr,
                     const C* fromptr,
                     const U* parents,
                     int64_t lenparents,
                     int64_t outlength,
                     int64_t* scan_in_array,
                     uint64_t invocation_index,
                     uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < lenparents) {
      toptr[parents[thread_id]] = scan_in_array[thread_id];
    }
  }
}
