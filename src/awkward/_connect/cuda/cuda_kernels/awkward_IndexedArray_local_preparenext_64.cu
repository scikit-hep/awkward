// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tocarry, starts, parents, parentslength, nextparents, nextlen, invocation_index, err_code) = args
//     if nextlen < 1024:
//         block_size = nextlen
//     else:
//         block_size = 1024
//     if block_size > 0:
//         grid_size1 = math.floor((nextlen + block_size - 1) / block_size)
//         grid_size2 = math.floor((parentslength + block[0] - 1) / block[0])
//     else:
//         grid_size1, grid_size2 = 1, 1
//     temp = cupy.zeros(nextlen, dtype=cupy.int64)
//     scan_in_array_nextstarts = cupy.zeros(len(starts) + 1, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_local_preparenext_64_a", tocarry.dtype, starts.dtype, parents.dtype, nextparents.dtype]))((grid_size1,), (block_size,), (tocarry, starts, parents, parentslength, nextparents, nextlen, temp, scan_in_array_nextstarts, len(starts), invocation_index, err_code))
//     scan_in_array_nextstarts = cupy.cumsum(scan_in_array_nextstarts)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_local_preparenext_64_b", tocarry.dtype, starts.dtype, parents.dtype, nextparents.dtype]))((grid_size2,), block, (tocarry, starts, parents, parentslength, nextparents, nextlen, temp, scan_in_array_nextstarts, len(starts), invocation_index, err_code))
// out["awkward_IndexedArray_local_preparenext_64_a", {dtype_specializations}] = None
// out["awkward_IndexedArray_local_preparenext_64_b", {dtype_specializations}] = None
// END PYTHON


template <typename T, typename C, typename U, typename V>
__global__ void
awkward_IndexedArray_local_preparenext_64_a(
    T* tocarry,
    const C* starts,
    const U* parents,
    const int64_t parentslength,
    const V* nextparents,
    const int64_t nextlen,
    int64_t* temp,
    int64_t* scan_in_array_nextstarts,
    int64_t startslength,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t idx = threadIdx.x;
    int64_t thread_id = blockIdx.x * blockDim.x + idx;

    if (thread_id < nextlen) {
      temp[thread_id] = 1;
    }
    __syncthreads();

    if (thread_id < nextlen) {
      for (int64_t stride = 1; stride < blockDim.x; stride *= 2) {
        int64_t val = 0;
        if (idx >= stride && thread_id < nextlen && nextparents[thread_id] == nextparents[thread_id - stride]) {
          val = temp[thread_id - stride];
        }
        __syncthreads();
        temp[thread_id] += val;
        __syncthreads();
      }

      int64_t nextparent = nextparents[thread_id];
      if (idx == blockDim.x - 1 || thread_id == nextlen - 1 || nextparents[thread_id] != nextparents[thread_id + 1]) {
        atomicAdd(&scan_in_array_nextstarts[nextparent + 1], temp[thread_id]);
      }
    }
  }
}

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_IndexedArray_local_preparenext_64_b(
    T* tocarry,
    const C* starts,
    const U* parents,
    const int64_t parentslength,
    const V* nextparents,
    const int64_t nextlen,
    int64_t* temp,
    int64_t* scan_in_array_nextstarts,
    int64_t startslength,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < parentslength) {
        int64_t parent = parents[thread_id];
        int64_t start = starts[parent];
        int64_t stop = 0;
        if (parent == startslength - 1) {
          stop = parentslength;
        } else {
          stop = starts[parent + 1];
        }

        if (start < stop) {
            int64_t next_start = scan_in_array_nextstarts[parent];
            int64_t next_stop = scan_in_array_nextstarts[parent + 1];

            if (next_start + (thread_id - start) < next_stop) {
                tocarry[thread_id] = next_start + (thread_id - start);
            } else {
                tocarry[thread_id] = -1;
            }
        } else {
            tocarry[thread_id] = -1;
        }
    }
  }
}
