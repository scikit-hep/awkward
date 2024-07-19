// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (outoffsets, outcarry, parents, lenparents, outlength, invocation_index, err_code) = args
//     if block[0] > 0:
//         grid_size = math.floor((lenparents + block[0] - 1) / block[0])
//     else:
//         grid_size = 1
//     temp = cupy.zeros(lenparents, dtype=cupy.int64)
//     scan_in_array = cupy.zeros(lenparents, dtype=cupy.int64)
//     scan_in_array_outoffsets = cupy.zeros(outlength + 1, dtype=outoffsets.dtype)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_RecordArray_reduce_nonlocal_outoffsets_64_a", outoffsets.dtype, outcarry.dtype, parents.dtype]))((grid_size,), block, (outoffsets, outcarry, parents, lenparents, outlength, temp, scan_in_array, scan_in_array_outoffsets, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_RecordArray_reduce_nonlocal_outoffsets_64_b", outoffsets.dtype, outcarry.dtype, parents.dtype]))((grid_size,), block, (outoffsets, outcarry, parents, lenparents, outlength, temp, scan_in_array, scan_in_array_outoffsets, invocation_index, err_code))
//     scan_in_array_outoffsets = cupy.cumsum(scan_in_array_outoffsets)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_RecordArray_reduce_nonlocal_outoffsets_64_c", outoffsets.dtype, outcarry.dtype, parents.dtype]))((grid_size,), block, (outoffsets, outcarry, parents, lenparents, outlength, temp, scan_in_array, scan_in_array_outoffsets, invocation_index, err_code))
// out["awkward_RecordArray_reduce_nonlocal_outoffsets_64_a", {dtype_specializations}] = None
// out["awkward_RecordArray_reduce_nonlocal_outoffsets_64_b", {dtype_specializations}] = None
// out["awkward_RecordArray_reduce_nonlocal_outoffsets_64_c", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_RecordArray_reduce_nonlocal_outoffsets_64_a(
    T* outoffsets,
    C* outcarry,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    T* temp,
    int64_t* scan_in_array,
    T* scan_in_array_outoffsets,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    scan_in_array_outoffsets[0] = 0;
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < lenparents - 1) {
      scan_in_array[0] = 1;
      if (parents[thread_id] != parents[thread_id + 1]) {
        scan_in_array[thread_id + 1] = 1;
      }
    }
    if (thread_id < outlength) {
      scan_in_array_outoffsets[thread_id + 1] = 0;
      outcarry[thread_id] = -1;
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_RecordArray_reduce_nonlocal_outoffsets_64_b(
    T* outoffsets,
    C* outcarry,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    T* temp,
    int64_t* scan_in_array,
    T* scan_in_array_outoffsets,
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
        atomicAdd(&scan_in_array_outoffsets[scan_in_array[thread_id]], temp[thread_id]);
      }
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_RecordArray_reduce_nonlocal_outoffsets_64_c(
    T* outoffsets,
    C* outcarry,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    T* temp,
    int64_t* scan_in_array,
    T* scan_in_array_outoffsets,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    outoffsets[0] = 0;
    if (thread_id < lenparents) {
      if (parents[thread_id] != parents[thread_id + 1]) {
        outcarry[parents[thread_id]] = scan_in_array[thread_id] - 1;
      }
    }
    if (thread_id < outlength) {
      if (outcarry[thread_id] == -1) {
        outcarry[thread_id] = lenparents > 0 && outlength > 1 ? atomicAdd(&scan_in_array[lenparents - 1], 1) : 0;
      }
      outoffsets[thread_id + 1] = (T)scan_in_array_outoffsets[thread_id + 1];
    }
  }
}
