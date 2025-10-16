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
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_argmin_a", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, atomic_toptr, temp, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_argmin_b", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, atomic_toptr, temp, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_reduce_argmin_c", cupy.dtype(toptr.dtype).type, cupy.dtype(fromptr.dtype).type, parents.dtype]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, atomic_toptr, temp, invocation_index, err_code))
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
    uint64_t* atomic_toptr,
    T* temp,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < outlength) {
      atomic_toptr[thread_id] = -1;
    }
  }
}

// Consistent tie-break: prefer lower index on ties (first occurrence)

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_argmin_b(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    uint64_t* atomic_toptr,
    T* temp,
    uint64_t invocation_index,
    uint64_t* err_code) {

  if (err_code[0] == NO_ERROR) {
    const uint64_t EMPTY = (uint64_t)(-1); // sentinel for "no winner yet"

    int64_t idx = threadIdx.x;
    int64_t thread_id = blockIdx.x * blockDim.x + idx;

    // Initialize temp with the global index for valid threads; keep out-of-range as -1
    if (thread_id < lenparents) {
      temp[thread_id] = thread_id;
    } else {
      if (thread_id < outlength) {
        temp[thread_id] = -1;
      }
    }
    __syncthreads();

    if (thread_id < lenparents) {
      // Intra-block tree reduction to compute block-local winner index in temp[thread_id]
      for (int64_t stride = 1; stride < blockDim.x; stride *= 2) {
        int64_t index = -1;
        if (idx >= stride && thread_id < lenparents && parents[thread_id] == parents[thread_id - stride]) {
          index = temp[thread_id - stride];
        }
        __syncthreads(); // ensure producers finished
        if (index != -1 && (temp[thread_id] == -1 ||
            fromptr[index] < fromptr[temp[thread_id]] ||
            (fromptr[index] == fromptr[temp[thread_id]] && index < temp[thread_id]))) {
          temp[thread_id] = index;
        }
        __syncthreads();
      }

      int64_t parent = parents[thread_id];
      // boundary thread for each parent emits the block-local candidate
      if (idx == blockDim.x - 1 || thread_id == lenparents - 1 || parents[thread_id] != parents[thread_id + 1]) {
        uint64_t candidate = (uint64_t) temp[thread_id];
        if (candidate != (uint64_t)-1) {
          // CAS loop: install or replace only when candidate is strictly better
          uint64_t cur = atomic_toptr[parent];
          while (true) {
            // If empty, try to install candidate directly
            if (cur == EMPTY) {
              uint64_t prev = atomicCAS(&atomic_toptr[parent], EMPTY, candidate);
              if (prev == EMPTY) {
                // installed successfully
                break;
              } else {
                // someone else wrote; update cur and re-evaluate
                cur = prev;
                continue;
              }
            } else {
              int64_t old_idx = (int64_t) cur;
              int64_t new_idx = (int64_t) candidate;

              C old_val = fromptr[old_idx];
              C new_val = fromptr[new_idx];

              // Candidate is better if new_val < old_val, or equal but lower index (first occurrence)
              if (new_val < old_val || (new_val == old_val && new_idx < old_idx)) {
                uint64_t prev = atomicCAS(&atomic_toptr[parent], cur, candidate);
                if (prev == cur) {
                  // replaced successfully
                  break;
                } else {
                  // lost race, refresh cur and retry
                  cur = prev;
                  continue;
                }
              } else {
                // stored winner is better (or equal with preferred index) -> done
                break;
              }
            }
          } // end CAS loop
        } // end candidate valid
      } // end boundary check
    } // end valid thread
  } // end err_code check
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_argmin_c(
    T* toptr,
    const C* fromptr,
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
