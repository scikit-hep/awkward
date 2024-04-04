// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (carrylen, slicestarts, slicestops, sliceouterlen, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(sliceouterlen, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_getitem_jagged_carrylen_a", carrylen.dtype, slicestarts.dtype, slicestops.dtype]))(grid, block, (carrylen, slicestarts, slicestops, sliceouterlen, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_getitem_jagged_carrylen_b", carrylen.dtype, slicestarts.dtype, slicestops.dtype]))(grid, block, (carrylen, slicestarts, slicestops, sliceouterlen, scan_in_array, invocation_index, err_code))
// out["awkward_ListArray_getitem_jagged_carrylen_a", {dtype_specializations}] = None
// out["awkward_ListArray_getitem_jagged_carrylen_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_ListArray_getitem_jagged_carrylen_a(
    T* carrylen,
    const C* slicestarts,
    const U* slicestops,
    int64_t sliceouterlen,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < sliceouterlen) {
      scan_in_array[thread_id] = (int64_t)(slicestops[thread_id] - slicestarts[thread_id]);
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_ListArray_getitem_jagged_carrylen_b(
    T* carrylen,
    const C* slicestarts,
    const U* slicestops,
    int64_t sliceouterlen,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    *carrylen = sliceouterlen > 0 ? scan_in_array[sliceouterlen - 1] : 0;
  }
}
