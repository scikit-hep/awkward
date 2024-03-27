// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tocarry, fromindex, length, invocation_index, err_code) = args
//     scan_in_array = cupy.empty(length, dtype=cupy.int64)
//     if length > 0:
//         scan_in_array = cupy.argsort(fromindex)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListOffsetArray_local_preparenext_64", tocarry.dtype, fromindex.dtype]))(grid, block, (tocarry, fromindex, length, scan_in_array, invocation_index, err_code))
// out["awkward_ListOffsetArray_local_preparenext_64", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C>
__global__ void
awkward_ListOffsetArray_local_preparenext_64(
    T* tocarry,
    const C* fromindex,
    int64_t length,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      tocarry[thread_id] = scan_in_array[thread_id];
    }
  }
}
