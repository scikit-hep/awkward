// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toindex, tolength, parents, parentslength, invocation_index, err_code) = args
//     scan_in_array_k = cupy.ones(parentslength, dtype=cupy.int64)
//     scan_in_array_j = cupy.zeros(parentslength, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_sorting_ranges_a', toindex.dtype, parents.dtype]))(grid, block, (toindex, tolength, parents, parentslength, scan_in_array_k, scan_in_array_j, invocation_index, err_code))
//     scan_in_array_k = cupy.cumsum(scan_in_array_k)
//     scan_in_array_j = cupy.cumsum(scan_in_array_j)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_sorting_ranges_b', toindex.dtype, parents.dtype]))(grid, block, (toindex, tolength, parents, parentslength, scan_in_array_k, scan_in_array_j, invocation_index, err_code))
// out["awkward_sorting_ranges_a", {dtype_specializations}] = None
// out["awkward_sorting_ranges_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C>
__global__ void
awkward_sorting_ranges_a(
    T* toindex,
    int64_t tolength,
    const C* parents,
    int64_t parentslength,
    int64_t* scan_in_array_k,
    int64_t* scan_in_array_j,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id > 0 && thread_id < parentslength) {
      if (parents[thread_id - 1] != parents[thread_id]) {
        scan_in_array_j[thread_id] = 1;
      }
    }
  }
}

template <typename T, typename C>
__global__ void
awkward_sorting_ranges_b(
    T* toindex,
    int64_t tolength,
    const C* parents,
    int64_t parentslength,
    int64_t* scan_in_array_k,
    int64_t* scan_in_array_j,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    toindex[0] = 0;
    if (thread_id > 0 && thread_id < parentslength) {
      if (parents[thread_id - 1] != parents[thread_id]) {
        toindex[scan_in_array_j[thread_id]] = scan_in_array_k[thread_id - 1];
      }
    }
    toindex[tolength - 1] = parentslength;
  }
}
