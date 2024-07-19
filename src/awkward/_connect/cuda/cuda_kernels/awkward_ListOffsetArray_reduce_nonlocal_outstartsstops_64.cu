// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (outstarts, outstops, distincts, lendistincts, outlength, invocation_index, err_code) = args
//     if block[0] > 0:
//         grid_size = math.floor((lendistincts + block[0] - 1) / block[0])
//     else:
//         grid_size = 1
//     temp = cupy.zeros(lendistincts, dtype=cupy.int64)
//     scan_in_array = cupy.zeros(outlength, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64_a", outstarts.dtype, outstops.dtype, distincts.dtype]))((grid_size,), block, (outstarts, outstops, distincts, lendistincts, outlength, temp, scan_in_array, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64_b", outstarts.dtype, outstops.dtype, distincts.dtype]))((grid_size,), block, (outstarts, outstops, distincts, lendistincts, outlength, temp, scan_in_array, invocation_index, err_code))
// out["awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64_a", {dtype_specializations}] = None
// out["awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64_a(
    T* outstarts,
    C* outstops,
    const U* distincts,
    int64_t lendistincts,
    int64_t outlength,
    int64_t* temp,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
      if (err_code[0] == NO_ERROR) {
        int64_t idx = threadIdx.x;
        int64_t thread_id = blockIdx.x * blockDim.x + idx;

        if (thread_id < lendistincts) {
          temp[thread_id] = (distincts[thread_id] != -1) ? 1 : 0;
        }
        __syncthreads();

        if (thread_id < lendistincts) {
          int64_t maxcount = lendistincts / outlength;
          int64_t id = thread_id / maxcount;

          for (int64_t stride = 1; stride < blockDim.x; stride *= 2) {
            int64_t val = 0;
            if (idx >= stride && thread_id < lendistincts && (thread_id / maxcount) == ((thread_id - stride) / maxcount)) {
              val = temp[thread_id - stride];
            }
            __syncthreads();
            temp[thread_id] += val;
            __syncthreads();
          }
          if (idx == blockDim.x - 1 || thread_id == lendistincts - 1 || (thread_id / maxcount) != ((thread_id + 1) / maxcount)) {
            atomicAdd(&scan_in_array[id], temp[thread_id]);
          }
        }
      }
}

template <typename T, typename C, typename U>
__global__ void
awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64_b(
    T* outstarts,
    C* outstops,
    const U* distincts,
    int64_t lendistincts,
    int64_t outlength,
    int64_t* temp,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if ((outlength > 0 && lendistincts > 0) ){
      if (thread_id < outlength) {
        int64_t maxcount = lendistincts / outlength;
        int64_t start = thread_id * maxcount;
        outstarts[thread_id] = start;
        outstops[thread_id] = start + scan_in_array[thread_id];
      }
    } else {
      if (thread_id < outlength) {
        outstarts[thread_id] = 0;
        outstops[thread_id] = 0;
      }
    }
  }
}
