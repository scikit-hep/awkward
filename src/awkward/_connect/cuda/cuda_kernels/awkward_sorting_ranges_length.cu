// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tolength, parents, parentslength, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(parentslength, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_sorting_ranges_length_a', tolength.dtype, parents.dtype]))(grid, block, (tolength, parents, parentslength, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(['awkward_sorting_ranges_length_b', tolength.dtype, parents.dtype]))(grid, block, (tolength, parents, parentslength, scan_in_array, invocation_index, err_code))
// out["awkward_sorting_ranges_length_a", {dtype_specializations}] = None
// out["awkward_sorting_ranges_length_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C>
__global__ void
awkward_sorting_ranges_length_a(
    T* tolength,
    const C* parents,
    int64_t parentslength,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < parentslength) {
      if (thread_id == 0 ) {
        scan_in_array[thread_id] = 2;
      }
      else {
        if (parents[thread_id - 1] != parents[thread_id]) {
          scan_in_array[thread_id] = 1;
        }
      }
    }
  }
}

template <typename T, typename C>
__global__ void
awkward_sorting_ranges_length_b(
    T* tolength,
    const C* parents,
    int64_t parentslength,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    *tolength = parentslength > 0 ? scan_in_array[parentslength - 1] : 2;
  }
}
