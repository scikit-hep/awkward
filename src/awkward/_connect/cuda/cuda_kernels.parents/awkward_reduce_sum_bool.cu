// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, offsets, outlength, invocation_index, err_code) = args
//     # Offsets-pipeline: derive parents from offsets+outlength so the kernel
//     # body (which still reads parents internally) runs unchanged.
//     lenparents = int(offsets[int(outlength)].item()) if int(outlength) >= 0 else 0
//     if int(outlength) > 0 and lenparents > 0:
//         counts = offsets[1:int(outlength) + 1] - offsets[:int(outlength)]
//         parents = cupy.repeat(cupy.arange(int(outlength), dtype=cupy.int64), counts.astype(cupy.int64))
//     else:
//         parents = cupy.zeros(0, dtype=cupy.int64)
//     if block[0] > 0:
//         grid_size = math.floor((lenparents + block[0] - 1) / block[0])
//     else:
//         grid_size = 1
//     temp = cupy.zeros(lenparents, dtype=toptr.dtype)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_sum_bool_a", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype, offsets.dtype]))((grid_size,), block, (toptr, fromptr, parents, offsets, lenparents, outlength, temp, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_sum_bool_b", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype, offsets.dtype]))((grid_size,), block, (toptr, fromptr, parents, offsets, lenparents, outlength, temp, invocation_index, err_code))
// out["awkward_reduce_sum_bool_a", {dtype_specializations}] = None
// out["awkward_reduce_sum_bool_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_reduce_sum_bool_a(
    T* toptr,
    const C* fromptr,
    const U* parents,
    const V* offsets,
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

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_reduce_sum_bool_b(
    T* toptr,
    const C* fromptr,
    const U* parents,
    const V* offsets,
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

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_reduce_sum_bool_c(
    T* toptr,
    const C* fromptr,
    const U* parents,
    const V* offsets,
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
