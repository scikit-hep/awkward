// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, parents, lenparents, outlength, identity, invocation_index, err_code) = args
//     if block[0] > 0:
//         grid_size = math.floor((lenparents + block[0] - 1) / block[0])
//     else:
//         grid_size = 1
//     temp = cupy.tile([identity, 0], lenparents)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_min_complex_a", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, toptr.dtype.type(identity), temp, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_min_complex_b", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, toptr.dtype.type(identity), temp, invocation_index, err_code))
// out["awkward_reduce_min_complex_a", {dtype_specializations}] = None
// out["awkward_reduce_min_complex_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_min_complex_a(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    T identity,
    T* temp,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < outlength) {
      toptr[thread_id * 2] = (T)identity;
      toptr[thread_id * 2 + 1] = (T)0;
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_min_complex_b(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    T identity,
    T* temp,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t idx = threadIdx.x;
    int64_t thread_id = blockIdx.x * blockDim.x + idx;
    if (thread_id < lenparents) {
      temp[thread_id * 2] = fromptr[thread_id * 2];
      temp[thread_id * 2 + 1] = fromptr[thread_id * 2 + 1];
    }
    __syncthreads();

    if (thread_id < lenparents) {
      for (int stride = 1; stride < blockDim.x; stride *= 2) {
        T stride_real = identity;
        T stride_imag = 0;
        if (idx >= stride && thread_id < lenparents && parents[thread_id] == parents[thread_id - stride]) {
          T current_real = temp[thread_id * 2];
          T current_imag = temp[thread_id * 2 + 1];
          stride_real = temp[(thread_id - stride) * 2];
          stride_imag = temp[(thread_id - stride) * 2 + 1];

          if (stride_real < current_real || (stride_real == current_real && stride_imag < current_imag)) {
            temp[thread_id * 2] = stride_real;
            temp[thread_id * 2 + 1] = stride_imag;
          }
        }
        __syncthreads();
      }

      int parent = parents[thread_id];
      if (idx == blockDim.x - 1 || thread_id == lenparents - 1 || parents[thread_id] != parents[thread_id + 1]) {
        atomicMinComplex(&toptr[parent * 2], &toptr[parent * 2 + 1], temp[thread_id * 2], temp[thread_id * 2 + 1]);
      }
    }
  }
}
