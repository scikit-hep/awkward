// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, parents, lenparents, outlength, invocation_index, err_code) = args
//     shared_mem_size = block[0] * toptr.dtype.itemsize
//     if block[0] > 0:
//         segment = math.floor((outlength + block[0] - 1) / block[0])
//         partial_size = outlength * ((lenparents + block[0] - 1) / block[0])
//     else:
//         segment = 0
//         partial_size = 0
//     partial = cupy.full(math.floor(partial_size), -1, dtype=toptr.dtype)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_argmin_a", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype]))(grid, block, (toptr, fromptr, parents, lenparents, outlength, partial, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_argmin_b", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype]))(grid, block, (toptr, fromptr, parents, lenparents, outlength, partial, invocation_index, err_code), shared_mem=shared_mem_size)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_argmin_c", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype]))((segment,), block, (toptr, fromptr, parents, lenparents, outlength, partial, invocation_index, err_code))
// out["awkward_reduce_argmin_a", {dtype_specializations}] = None
// out["awkward_reduce_argmin_b", {dtype_specializations}] = None
// out["awkward_reduce_argmin_c", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_argmin_a(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    T* partial,
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
    T* partial,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory[];
    T *shared_mem = reinterpret_cast<T *>(shared_memory);

    int64_t idx = threadIdx.x;
    int64_t thread_id = blockIdx.x * blockDim.x + idx;

    if (thread_id < lenparents) {
      shared_mem[idx] = thread_id;
    } else {
      shared_mem[idx] = -1;
    }
    __syncthreads();

    for (int64_t stride = 1; stride < blockDim.x; stride *= 2) {
      int64_t index = -1;
      if (idx >= stride && thread_id < lenparents && parents[thread_id] == parents[thread_id - stride]) {
        index = shared_mem[idx - stride];
      }
      if (index != -1 && (shared_mem[idx] == -1 || fromptr[index] < fromptr[shared_mem[idx]] ||
         (fromptr[index] == fromptr[shared_mem[idx]] && index < shared_mem[idx]))) {
        shared_mem[idx] = index;
      }
      __syncthreads();
    }

    if (thread_id < lenparents) {
      int64_t parent = parents[thread_id];
      if (idx == blockDim.x - 1 || thread_id == lenparents - 1 || parents[thread_id] != parents[thread_id + 1]) {
        partial[blockIdx.x * outlength + parent] = shared_mem[idx];
      }
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_argmin_c(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    T* partial,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < outlength) {
      int64_t argmin = -1;
      int64_t blocks = (lenparents + blockDim.x - 1) / blockDim.x;
      for (int64_t i = 0; i < blocks; ++i) {
        int64_t index = partial[i * outlength + thread_id];
        if (index != -1 && (argmin == -1 || fromptr[index] < fromptr[argmin]) ||
           (fromptr[index] == fromptr[argmin] && index < argmin)) {
          argmin = index;
        }
      }
      toptr[thread_id] = argmin;
    }
  }
}
