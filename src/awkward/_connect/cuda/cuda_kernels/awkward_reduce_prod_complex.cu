// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, parents, offsets, lenparents, outlength, invocation_index, err_code) = args
//     if block[0] > 0:
//         grid_size = math.floor((lenparents + block[0] - 1) / block[0])
//     else:
//         grid_size = 1
//     temp = cupy.zeros(2 * grid_size * block[0], dtype=toptr.dtype)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_prod_complex_a", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype, offsets.dtype]))((grid_size,), block, (toptr, fromptr, parents, offsets, lenparents, outlength, temp, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_prod_complex_b", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype, offsets.dtype]))((grid_size,), block, (toptr, fromptr, parents, offsets, lenparents, outlength, temp, invocation_index, err_code))
// out["awkward_reduce_prod_complex_a", {dtype_specializations}] = None
// out["awkward_reduce_prod_complex_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_reduce_prod_complex_a(
    T* toptr,
    const C* fromptr,
    const U* parents,
    const V* offsets,
    int64_t lenparents,
    int64_t outlength,
    T* temp,                 // temp is a global buffer sized 2*lenparents, pre-filled with (1,0)
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // initialize output slots to 1 + 0j (neutral element for product)
    if (thread_id < outlength) {
      toptr[thread_id * 2]     = (T)1; // real
      toptr[thread_id * 2 + 1] = (T)0; // imag
    }
  }
}

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_reduce_prod_complex_b(
    T* toptr,
    const C* fromptr,
    const U* parents,
    const V* offsets,
    int64_t lenparents,
    int64_t outlength,
    T* temp,                 // global workspace: length 2*lenparents (real, imag pairs)
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {

    int64_t idx = threadIdx.x;
    int64_t thread_id = blockIdx.x * blockDim.x + idx;

    // load input into temp (real, imag)
    if (thread_id < lenparents) {
      temp[thread_id * 2]     = fromptr[thread_id * 2];
      temp[thread_id * 2 + 1] = fromptr[thread_id * 2 + 1];
    }
    // ensure other threads don't read uninitialized entries in the block
    __syncthreads();

    // Perform in-block tree reduction across threads, but only merge elements
    // that share the same parent. We'll reduce so that the rightmost thread
    // in each run of equal-parents holds the product of that run.
    for (int64_t stride = 1; stride < blockDim.x; stride *= 2) {
      T right_r = (T)1;
      T right_i = (T)0;

      if (thread_id < lenparents) {
        int64_t partner = thread_id - stride;
        if (idx >= stride && partner >= 0 && partner < lenparents &&
            parents[thread_id] == parents[partner]) {
          right_r = temp[partner * 2];
          right_i = temp[partner * 2 + 1];
        }
      }
      __syncthreads();

      if (thread_id < lenparents) {
        T left_r = temp[thread_id * 2];
        T left_i = temp[thread_id * 2 + 1];
        // complex multiply: (left) * (right)
        T res_r = left_r * right_r - left_i * right_i;
        T res_i = left_r * right_i + left_i * right_r;
        temp[thread_id * 2]     = res_r;
        temp[thread_id * 2 + 1] = res_i;
      }
      __syncthreads();
    }

    if (thread_id < lenparents) {
      int64_t parent = parents[thread_id];
      if (idx == blockDim.x - 1 || thread_id == lenparents - 1 || parents[thread_id] != parents[thread_id + 1]) {
        T cand_r = temp[thread_id * 2];
        T cand_i = temp[thread_id * 2 + 1];
        // atomic multiplication into toptr[parent*2 .. parent*2+1]
        // assume atomicMulComplex(&real, &imag, cand_real, cand_imag) exists and
        // does an atomic read-modify-write multiply in double precision, preserving
        // IEEE semantics (NaN/Inf propagation).
        atomicMulComplex(&toptr[parent * 2], &toptr[parent * 2 + 1], cand_r, cand_i);
      }
    }
  }
}
