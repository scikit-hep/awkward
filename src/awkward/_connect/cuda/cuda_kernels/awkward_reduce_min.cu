// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     """
//     Min reduction for sorted, present parents on device:
//     (toptr, fromptr, parents, lenparents, outlength, identity, invocation_index, err_code)
//     """
//     (toptr, fromptr, parents, lenparents,
//      outlength, identity, invocation_index, err_code) = args
//     if block[0] > 0:
//         grid_size = math.floor((lenparents + block[0] - 1) / block[0])
//     else:
//         grid_size = 1
//     # atomic_toptr initialized to identity
//     atomic_toptr = cupy.full(outlength, identity, dtype=fromptr.dtype)
//     temp = cupy.zeros(lenparents, dtype=fromptr.dtype)
//     cuda_kernel_templates.get_function(
//         fetch_specialization([
//             "awkward_reduce_min_a",
//             cupy.dtype(toptr.dtype).type,
//             cupy.dtype(fromptr.dtype).type,
//             parents.dtype
//         ])
//     )((grid_size,), block,
//       (toptr, fromptr, parents, lenparents, outlength,
//        atomic_toptr, temp, identity, invocation_index, err_code))
//     cuda_kernel_templates.get_function(
//         fetch_specialization([
//             "awkward_reduce_min_b",
//             cupy.dtype(toptr.dtype).type,
//             cupy.dtype(fromptr.dtype).type,
//             parents.dtype
//         ])
//     )((grid_size,), block,
//       (toptr, fromptr, parents, lenparents, outlength,
//        atomic_toptr, temp, identity, invocation_index, err_code))
//     cuda_kernel_templates.get_function(
//         fetch_specialization([
//             "awkward_reduce_min_c",
//             cupy.dtype(toptr.dtype).type,
//             cupy.dtype(fromptr.dtype).type,
//             parents.dtype
//         ])
//     )((grid_size,), block,
//       (toptr, fromptr, parents, lenparents, outlength,
//        atomic_toptr, temp, identity, invocation_index, err_code))
// out["awkward_reduce_min_a", {dtype_specializations}] = None
// out["awkward_reduce_min_b", {dtype_specializations}] = None
// out["awkward_reduce_min_c", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_min_a(
    C* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    C* atomic_toptr,
    C* temp,
    C identity,
    uint64_t invocation_index,
    uint64_t* err_code) {

  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < outlength) {
      atomic_toptr[thread_id] = identity;
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_min_b(
    T* toptr,                  // (unused; kept for signature compatibility)
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    C* atomic_toptr,
    C* temp,
    C identity,
    uint64_t invocation_index,
    uint64_t* err_code) {

  if (err_code[0] == NO_ERROR) {
    const C INIT = identity;

    int64_t idx = threadIdx.x;
    int64_t thread_id = blockIdx.x * blockDim.x + idx;

    if (thread_id < lenparents) {
      temp[thread_id] = fromptr[thread_id];
    } else {
      // make sure out-of-range lanes don't affect reduction
      // (INIT should be +inf or neutral for min)
      // Only needed if this thread participates in syncthreads below
      // Safe to write; temp is size lenparents
    }
    __syncthreads();

    if (thread_id < lenparents) {
      // segmented in-block reduction by parent id
      for (int64_t stride = 1; stride < blockDim.x; stride *= 2) {
        C other = INIT;
        if (idx >= stride &&
            thread_id - stride >= 0 &&
            parents[thread_id] == parents[thread_id - stride]) {
          other = temp[thread_id - stride];
        }
        __syncthreads();

        if (other < temp[thread_id]) {
          temp[thread_id] = other;
        }
        __syncthreads();
      }

      int64_t parent = parents[thread_id];
      // boundary thread for each parent emits its block-local candidate
      if (idx == blockDim.x - 1 ||
          thread_id == lenparents - 1 ||
          parents[thread_id] != parents[thread_id + 1]) {
        C candidate = temp[thread_id];
        // atomically fold candidate into the per-parent global slot
        atomicMin(&atomic_toptr[parent], candidate);
      }
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_min_c(
    C* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    C* atomic_toptr,
    C* temp,
    C identity,
    uint64_t invocation_index,
    uint64_t* err_code) {

  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < outlength) {
      toptr[thread_id] = atomic_toptr[thread_id];
    }
  }
}
