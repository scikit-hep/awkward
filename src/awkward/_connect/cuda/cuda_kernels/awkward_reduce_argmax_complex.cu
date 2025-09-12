// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, parents, lenparents, outlength, invocation_index, err_code) = args
//     if block[0] > 0:
//         grid_size = math.floor((lenparents + block[0] - 1) / block[0])
//     else:
//         grid_size = 1
//     atomic_toptr = cupy.array(toptr, dtype=cupy.uint64)
//     temp = cupy.zeros(lenparents, dtype=toptr.dtype)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_argmax_complex_a", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, atomic_toptr, temp, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_argmax_complex_b", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, atomic_toptr, temp, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_argmax_complex_c", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, atomic_toptr, temp, invocation_index, err_code))
// out["awkward_reduce_argmax_complex_a", {dtype_specializations}] = None
// out["awkward_reduce_argmax_complex_b", {dtype_specializations}] = None
// out["awkward_reduce_argmax_complex_c", {dtype_specializations}] = None
// END PYTHON

// argmax: lexicographic compare (real, then imag), tie-breaker: lower index
template <typename T, typename C, typename U>
__global__ void
awkward_reduce_argmax_complex_a(
    T* toptr,
    const C* fromptr,   // [real, imag] pairs
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    uint64_t* atomic_toptr,
    T* temp,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < outlength) {
      atomic_toptr[thread_id] = (uint64_t)(-1);
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_argmax_complex_b(
    T* toptr,
    const C* fromptr,   // flattened [real, imag] array
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    uint64_t* atomic_toptr,
    T* temp,
    uint64_t invocation_index,
    uint64_t* err_code) {

  if (err_code[0] == NO_ERROR) {
    const uint64_t EMPTY = (uint64_t)(-1);

    int64_t idx = threadIdx.x;
    int64_t thread_id = blockIdx.x * blockDim.x + idx;

    if (thread_id < lenparents) {
      temp[thread_id] = thread_id;
    } else if (thread_id < outlength) {
      temp[thread_id] = -1;
    }
    __syncthreads();

    if (thread_id < lenparents) {
      // intra-block reduction
      for (int64_t stride = 1; stride < blockDim.x; stride *= 2) {
        int64_t index = -1;
        if (idx >= stride && parents[thread_id] == parents[thread_id - stride]) {
          index = temp[thread_id - stride];
        }
        __syncthreads();

        if (index != -1) {
          // load old candidate
          int64_t old_idx = temp[thread_id];
          double old_real = (double)fromptr[2 * old_idx];
          double old_imag = (double)fromptr[2 * old_idx + 1];
          // load new candidate
          double new_real = (double)fromptr[2 * index];
          double new_imag = (double)fromptr[2 * index + 1];

          // NUMPY LEXICOGRAPHIC (real, then imag), prefer lower index on ties
          bool better = (new_real > old_real) ||
                        (new_real == old_real && new_imag > old_imag) ||
                        (new_real == old_real && new_imag == old_imag && index < old_idx);

          if (old_idx == -1 || better) {
            temp[thread_id] = index;
          }
        }
        __syncthreads();
      }

      int64_t parent = parents[thread_id];
      if (idx == blockDim.x - 1 || thread_id == lenparents - 1 ||
          parents[thread_id] != parents[thread_id + 1]) {
        uint64_t candidate = (uint64_t)temp[thread_id];
        if (candidate != (uint64_t)-1) {
          uint64_t cur = atomic_toptr[parent];
          while (true) {
            if (cur == EMPTY) {
              uint64_t prev = atomicCAS(&atomic_toptr[parent], EMPTY, candidate);
              if (prev == EMPTY) break;
              cur = prev;
              continue;
            } else {
              int64_t old_idx = (int64_t)cur;
              int64_t new_idx = (int64_t)candidate;

              double old_real = (double)fromptr[2 * old_idx];
              double old_imag = (double)fromptr[2 * old_idx + 1];
              double new_real = (double)fromptr[2 * new_idx];
              double new_imag = (double)fromptr[2 * new_idx + 1];

              bool better = (new_real > old_real) ||
                            (new_real == old_real && new_imag > old_imag) ||
                            (new_real == old_real && new_imag == old_imag && new_idx < old_idx);

              if (better) {
                uint64_t prev = atomicCAS(&atomic_toptr[parent], cur, candidate);
                if (prev == cur) break;
                cur = prev;
                continue;
              } else {
                break;
              }
            }
          }
        }
      }
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_argmax_complex_c(
    T* toptr,
    const C* fromptr,   // unused
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    uint64_t* atomic_toptr,
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
