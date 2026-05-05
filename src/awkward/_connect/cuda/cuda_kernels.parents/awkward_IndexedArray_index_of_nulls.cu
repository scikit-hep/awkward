// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toindex, fromindex, offsets, outlength, starts, invocation_index, err_code) = args
//     # In the offsets-pipeline, "lenindex" = total elements = offsets[outlength].
//     lenindex = int(offsets[int(outlength)].item())
//     # Derive parents from offsets so the kernel body (still parents-driven)
//     # runs unchanged.
//     if int(outlength) > 0 and lenindex > 0:
//         counts = offsets[1:int(outlength) + 1] - offsets[:int(outlength)]
//         parents = cupy.repeat(cupy.arange(int(outlength), dtype=cupy.int64), counts.astype(cupy.int64))
//     else:
//         parents = cupy.zeros(0, dtype=cupy.int64)
//     scan_in_array = cupy.zeros(lenindex, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_index_of_nulls_a", toindex.dtype, fromindex.dtype, parents.dtype, starts.dtype]))(grid, block, (toindex, fromindex, lenindex, parents, starts, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_index_of_nulls_b", toindex.dtype, fromindex.dtype, parents.dtype, starts.dtype]))(grid, block, (toindex, fromindex, lenindex, parents, starts, scan_in_array, invocation_index, err_code))
// out["awkward_IndexedArray_index_of_nulls_a", {dtype_specializations}] = None
// out["awkward_IndexedArray_index_of_nulls_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_IndexedArray_index_of_nulls_a(
    T* toindex,
    const C* fromindex,
    int64_t lenindex,
    const U* parents,
    const V* starts,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < lenindex) {
      if (fromindex[thread_id] < 0) {
        scan_in_array[thread_id] = 1;
      }
    }
  }
}

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_IndexedArray_index_of_nulls_b(
    T* toindex,
    const C* fromindex,
    int64_t lenindex,
    const U* parents,
    const V* starts,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < lenindex) {
      if (fromindex[thread_id] < 0) {
        int64_t parent = parents[thread_id];
        int64_t start = starts[parent];
        toindex[scan_in_array[thread_id] - 1] = thread_id - start;
      }
    }
  }
}
