// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, parents, lenparents, outlength, invocation_index, err_code) = args
//     atomicAdd_toptr = cupy.array(toptr, dtype=cupy.uint64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_count_64_a", toptr.dtype, parents.dtype]))(grid, block, (toptr, fromptr, parents, lenparents, outlength, atomicAdd_toptr, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_count_64_b", toptr.dtype, parents.dtype]))(grid, block, (toptr, fromptr, parents, lenparents, outlength, atomicAdd_toptr, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_count_64_c", toptr.dtype, parents.dtype]))(grid, block, (toptr, fromptr, parents, lenparents, outlength, atomicAdd_toptr, invocation_index, err_code))
// out["awkward_reduce_count_64_a", {dtype_specializations}] = None
// out["awkward_reduce_count_64_b", {dtype_specializations}] = None
// out["awkward_reduce_count_64_c", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename U>
__global__ void
awkward_reduce_count_64_a(
    T* toptr,
    const bool* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    uint64_t* atomicAdd_toptr,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < outlength) {
      atomicAdd_toptr[thread_id] = 0;
    }
  }
}

template <typename T, typename U>
__global__ void
awkward_reduce_count_64_b(
    T* toptr,
    const bool* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    uint64_t* atomicAdd_toptr,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < lenparents) {
      atomicAdd(atomicAdd_toptr + parents[thread_id], (uint64_t)1);
    }
  }
}

template <typename T, typename U>
__global__ void
awkward_reduce_count_64_c(T* toptr,
                          const bool* fromptr,
                          const U* parents,
                          int64_t lenparents,
                          int64_t outlength,
                          uint64_t* atomicAdd_toptr,
                          uint64_t invocation_index,
                          uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < outlength) {
      toptr[thread_id] = (T)atomicAdd_toptr[thread_id];
    }
  }
}
