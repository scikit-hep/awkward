// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, parents, lenparents, outlength, invocation_index, err_code) = args
//     if block[0] > 0:
//         segment = math.floor((outlength + block[0] - 1) / block[0])
//         grid_size = math.floor((lenparents + block[0] - 1) / block[0])
//     else:
//         segment = 0
//         grid_size = 1
//     partial = cupy.full(outlength * grid_size, -1, dtype=toptr.dtype)
//     temp = cupy.zeros(lenparents, dtype=toptr.dtype)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_argmax_a", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, partial, temp, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_argmax_b", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, partial, temp, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_argmax_c", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype]))((segment,), block, (toptr, fromptr, parents, lenparents, outlength, partial, temp, invocation_index, err_code))
// out["awkward_reduce_argmax_a", {dtype_specializations}] = None
// out["awkward_reduce_argmax_b", {dtype_specializations}] = None
// out["awkward_reduce_argmax_c", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_argmax_a(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    T* partial,
    T* temp,
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
awkward_reduce_argmax_b(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    T* partial,
    T* temp,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t idx = threadIdx.x;
    int64_t thread_id = blockIdx.x * blockDim.x + idx;

    if (thread_id < lenparents) {
      temp[idx] = thread_id;
    }
    __syncthreads();

    for (int64_t stride = 1; stride < blockDim.x; stride *= 2) {
      int64_t index = -1;
      if (idx >= stride && thread_id < lenparents && parents[thread_id] == parents[thread_id - stride]) {
        index = temp[idx - stride];
      }
      if (index != -1 && (temp[idx] == -1 || fromptr[index] > fromptr[temp[idx]] ||
         (fromptr[index] == fromptr[temp[idx]] && index < temp[idx]))) {
        temp[idx] = index;
      }
      __syncthreads();
    }

    if (thread_id < lenparents) {
      int64_t parent = parents[thread_id];
      if (idx == blockDim.x - 1 || thread_id == lenparents - 1 || parents[thread_id] != parents[thread_id + 1]) {
        partial[blockIdx.x * outlength + parent] = temp[idx];
      }
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_argmax_c(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    T* partial,
    T* temp,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < outlength) {
      int64_t argmax = -1;
      int64_t blocks = (lenparents + blockDim.x - 1) / blockDim.x;
      for (int64_t i = 0; i < blocks; ++i) {
        int64_t index = partial[i * outlength + thread_id];
        if (index != -1 && (argmax == -1 || fromptr[index] > fromptr[argmax]) ||
           (fromptr[index] == fromptr[argmax] && index < argmax)) {
          argmax = index;
        }
      }
      toptr[thread_id] = argmax;
    }
  }
}
