// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, parents, lenparents, outlength, invocation_index, err_code) = args
//     if block[0] > 0:
//         grid_size = math.floor((lenparents + block[0] - 1) / block[0])
//     else:
//         grid_size = 1
//     atomic_toptr = cupy.array(toptr, dtype=cupy.uint32)
//     temp = cupy.ones(lenparents, dtype=toptr.dtype)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_prod_bool_a", bool_, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, atomic_toptr, temp, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_prod_bool_b", bool_, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, atomic_toptr, temp, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_prod_bool_c", bool_, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, atomic_toptr, temp, invocation_index, err_code))
// out["awkward_reduce_prod_bool_a", {dtype_specializations}] = None
// out["awkward_reduce_prod_bool_b", {dtype_specializations}] = None
// out["awkward_reduce_prod_bool_c", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_prod_bool_a(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    uint32_t* atomic_toptr,
    T* temp,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < outlength) {
      atomic_toptr[thread_id] = 1;
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_prod_bool_b(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    uint32_t* atomic_toptr,
    T* temp,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t idx = threadIdx.x;
    int64_t thread_id = blockIdx.x * blockDim.x + idx;

    if (thread_id < lenparents) {
      temp[thread_id] = fromptr[thread_id];
    }
    __syncthreads();

    if (thread_id < lenparents) {
      for (int64_t stride = 1; stride < blockDim.x; stride *= 2) {
        T val = 1;
        if (idx >= stride && thread_id < lenparents && parents[thread_id] == parents[thread_id - stride]) {
          val = temp[thread_id - stride];
        }
        __syncthreads();
        temp[thread_id] &= (val != 0);
        __syncthreads();
      }

      int64_t parent = parents[thread_id];
      if (idx == blockDim.x - 1 || thread_id == lenparents - 1 || parents[thread_id] != parents[thread_id + 1]) {
        atomicAnd(&atomic_toptr[parent], temp[thread_id]);
      }
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_prod_bool_c(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    uint32_t* atomic_toptr,
    T* temp,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < outlength) {
      toptr[thread_id] = (T)(atomic_toptr[thread_id]);
    }
  }
}
