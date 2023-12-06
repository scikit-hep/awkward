// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, parents, lenparents,outlength, invocation_index,err_code) = args
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_max_complex_a", toptr.dtype, fromptr.dtype, parents.dtype]))(grid, block, (toptr, fromptr, parents, lenparents,outlength, invocation_index,err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_max_complex_b", toptr.dtype, fromptr.dtype, parents.dtype]))(grid, block, (toptr, fromptr, parents, lenparents,outlength, invocation_index,err_code))
// out["awkward_reduce_max_complex_a", {dtype_specializations}] = None
// out["awkward_reduce_max_complex_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_max_complex_a(T* toptr,
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
      toptr[thread_id * 2] = identity;
      toptr[thread_id * 2 + 1] = 0;
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_max_complex_b(T* toptr,
                     const C* fromptr,
                     const U* parents,
                     int64_t lenparents,
                     int64_t outlength,
                     T identity,
                     uint64_t invocation_index,
                     uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < lenparents) {
      int64_t parent = parents[thread_id];
      C x = fromptr[thread_id * 2];
      C y = fromptr[thread_id * 2 + 1];
      if (x > toptr[parent * 2]  ||
        (x == toptr[parent * 2]  &&  y > toptr[parent * 2 + 1])) {
        toptr[parent * 2] = x;
        toptr[parent * 2 + 1] = y;
      }
    }
  }
}
