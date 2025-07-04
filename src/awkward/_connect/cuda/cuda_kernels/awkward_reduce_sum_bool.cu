// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, parents, lenparents, outlength, invocation_index, err_code) = args
//     if outlength == 0:
//         return  # Nothing to do, skip the rest
//     block_size = min(outlength, 1024)
//     grid_size = max(1, math.ceil((max(lenparents, outlength) + block_size - 1) / block_size))
//     atomic_toptr = cupy.array(toptr, dtype=cupy.uint32)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_sum_bool_a", bool_, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), (block_size,), (toptr, fromptr, parents, lenparents, outlength, atomic_toptr, toptr, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_sum_bool_b", bool_, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), (block_size,), (toptr, fromptr, parents, lenparents, outlength, atomic_toptr, toptr, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_sum_bool_c", bool_, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), (block_size,), (toptr, fromptr, parents, lenparents, outlength, atomic_toptr, toptr, invocation_index, err_code))
// out["awkward_reduce_sum_bool_a", {dtype_specializations}] = None
// out["awkward_reduce_sum_bool_b", {dtype_specializations}] = None
// out["awkward_reduce_sum_bool_c", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_sum_bool_a(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    uint32_t* atomic_toptr,
    T* /*unused*/,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < outlength) {
      atomic_toptr[thread_id] = 0;
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_sum_bool_b(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    uint32_t* atomic_toptr,
    T* /*unused*/,
    uint64_t invocation_index,
    uint64_t* err_code) {

  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < lenparents) {
      T val = (fromptr[thread_id] != 0) ? 1 : 0;
      int64_t parent = parents[thread_id];
      if (parent >= 0 && parent < outlength) {
        atomicOr(&atomic_toptr[parent], val);
      }
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_sum_bool_c(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    uint32_t* atomic_toptr,
    T* /*unused*/,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < outlength) {
      toptr[thread_id] = (T)(atomic_toptr[thread_id]);
    }
  }
}
