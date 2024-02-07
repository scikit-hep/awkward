// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, parents, lenparents, outlength, invocation_index, err_code) = args
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_argmin_a", toptr.dtype, fromptr.dtype, parents.dtype]))(grid, block, (toptr, fromptr, parents, lenparents, outlength, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_argmin_b", toptr.dtype, fromptr.dtype, parents.dtype]))(grid, block, (toptr, fromptr, parents, lenparents, outlength, invocation_index, err_code))
// out["awkward_reduce_argmin_a", {dtype_specializations}] = None
// out["awkward_reduce_argmin_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_argmin_a(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < outlength) {
      toptr[thread_id] = -1;
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_argmin_b(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < lenparents) {
      int64_t parent = parents[thread_id];
      if (toptr[parent] == -1 ||
          (fromptr[thread_id] < (fromptr[toptr[parent]]))) {
        toptr[parent] = thread_id;
      }
    }
  }
}
