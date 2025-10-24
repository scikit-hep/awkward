// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, parents, lenparents, outlength, invocation_index, err_code) = args
//     if block[0] > 0:
//         grid_size = math.floor((lenparents + block[0] - 1) / block[0])
//     else:
//         grid_size = 1
//     temp = cupy.tile([1, 0], lenparents)
//     temp = temp.astype(cupy.dtype(toptr.dtype))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_prod_complex_a", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, temp, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_prod_complex_b", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, temp, invocation_index, err_code))
// out["awkward_reduce_prod_complex_a", {dtype_specializations}] = None
// out["awkward_reduce_prod_complex_b", {dtype_specializations}] = None
// END PYTHON


// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_prod_complex_a(
    T* toptr,
    const C* fromptr,
    const U* parents,
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

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_prod_complex_b(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    T* temp,                 // global workspace: length 2*lenparents (real, imag pairs)
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {

    // global indices and thread-local index
    int64_t base_thread = blockIdx.x * blockDim.x;
    int64_t idx = threadIdx.x;
    int64_t thread_id = base_thread + idx;

    // load input into temp (real, imag)
    if (thread_id < lenparents) {
      temp[thread_id * 2]     = fromptr[thread_id * 2];
      temp[thread_id * 2 + 1] = fromptr[thread_id * 2 + 1];
    }
    // ensure other threads don't read uninitialized entries in the block
    __syncthreads();

    if (thread_id < lenparents) {
      // Perform in-block tree reduction across threads, but only merge elements
      // that share the same parent. We'll reduce so that the rightmost thread
      // in each run of equal-parents holds the product of that run.
      for (int64_t stride = 1; stride < blockDim.x; stride *= 2) {
        // compute the partner index (global)
        int64_t partner_id = thread_id - stride;

        // default: no partner -> keep own value
        double left_r = (double)temp[thread_id * 2];
        double left_i = (double)temp[thread_id * 2 + 1];
        double res_r  = left_r;
        double res_i  = left_i;

        // if this thread has a valid partner in the block and the partner
        // has the same parent, then multiply own value by partner's value
        if (idx >= stride && partner_id >= 0 &&
            parents[thread_id] == parents[partner_id]) {

          double right_r = (double)temp[partner_id * 2];
          double right_i = (double)temp[partner_id * 2 + 1];

          // complex multiply: (a+ib) * (c+id) = (a*c - b*d) + i(a*d + b*c)
          // Here we compute left * right
          res_r = left_r * right_r - left_i * right_i;
          res_i = left_r * right_i + left_i * right_r;
        }

        // barrier to make sure all reads are done before writes
        __syncthreads();

        // write back reduced value
        temp[thread_id * 2]     = (T)res_r;
        temp[thread_id * 2 + 1] = (T)res_i;

        // sync before next stride
        __syncthreads();
      }

      // after reduction, the thread that is the end of a parent-run (or last in block)
      // should push its candidate into the global toptr using atomic multiply
      int64_t parent = parents[thread_id];
      bool is_end_of_run = (idx == blockDim.x - 1) ||
                           (thread_id == lenparents - 1) ||
                           (parents[thread_id] != parents[thread_id + 1]);

      if (is_end_of_run) {
        // candidate is the reduced product for this run
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
