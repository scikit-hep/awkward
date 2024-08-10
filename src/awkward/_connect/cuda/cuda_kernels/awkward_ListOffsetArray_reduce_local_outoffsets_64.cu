// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (outoffsets, parents, lenparents, outlength, invocation_index, err_code) = args
//     if block[0] > 0:
//         grid_size = math.floor((lenparents + block[0] - 1) / block[0])
//     else:
//         grid_size = 1
//     temp = cupy.zeros(lenparents, dtype=cupy.int64)
//     scan_in_array = cupy.zeros(outlength, dtype=cupy.uint64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_reduce_local_outoffsets_64_a", cupy.dtype(outoffsets.dtype).type, parents.dtype]))((grid_size,), block, (outoffsets, parents, lenparents, outlength, scan_in_array, temp, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_reduce_local_outoffsets_64_b", cupy.dtype(outoffsets.dtype).type, parents.dtype]))((grid_size,), block, (outoffsets, parents, lenparents, outlength, scan_in_array, temp, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_reduce_local_outoffsets_64_c", cupy.dtype(outoffsets.dtype).type, parents.dtype]))((grid_size,), block, (outoffsets, parents, lenparents, outlength, scan_in_array, temp, invocation_index, err_code))
// out["awkward_ListOffsetArray_reduce_local_outoffsets_64_a", {dtype_specializations}] = None
// out["awkward_ListOffsetArray_reduce_local_outoffsets_64_b", {dtype_specializations}] = None
// out["awkward_ListOffsetArray_reduce_local_outoffsets_64_c", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C>
__global__ void
awkward_ListOffsetArray_reduce_local_outoffsets_64_a(
    T* outoffsets,
    const C* parents,
    int64_t lenparents,
    int64_t outlength,
    uint64_t* scan_in_array,
    int64_t* temp,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < outlength) {
      outoffsets[thread_id] = 0;
    }
  }
}

template <typename T, typename C>
__global__ void
awkward_ListOffsetArray_reduce_local_outoffsets_64_b(
    T* outoffsets,
    const C* parents,
    int64_t lenparents,
    int64_t outlength,
    uint64_t* scan_in_array,
    int64_t* temp,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t idx = threadIdx.x;
    int64_t thread_id = blockIdx.x * blockDim.x + idx;

    if (thread_id < lenparents) {
      temp[thread_id] = 1;
    }
    __syncthreads();

    if (thread_id < lenparents) {
      for (int64_t stride = 1; stride < blockDim.x; stride *= 2) {
        int64_t val = 0;
        if (idx >= stride && thread_id < lenparents && parents[thread_id] == parents[thread_id - stride]) {
          val = temp[thread_id - stride];
        }
        __syncthreads();
        temp[thread_id] += val;
        __syncthreads();
      }

      int64_t parent = parents[thread_id];
      if (idx == blockDim.x - 1 || thread_id == lenparents - 1 || parents[thread_id] != parents[thread_id + 1]) {
        atomicAdd(&scan_in_array[parent], temp[thread_id]);
      }
    }
  }
}

template <typename T, typename C>
__global__ void
awkward_ListOffsetArray_reduce_local_outoffsets_64_c(
    T* outoffsets,
    const C* parents,
    int64_t lenparents,
    int64_t outlength,
    uint64_t* scan_in_array,
    int64_t* temp,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    outoffsets[0] = 0;

    if (thread_id < outlength) {
      outoffsets[thread_id + 1] = (T)(scan_in_array[thread_id]);
    }
  }
}
