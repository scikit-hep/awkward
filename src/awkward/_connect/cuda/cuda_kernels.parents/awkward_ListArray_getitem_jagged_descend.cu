// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tooffsets, slicestarts, slicestops, sliceouterlen, fromstarts, fromstops, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(sliceouterlen, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_getitem_jagged_descend_a", tooffsets.dtype, slicestarts.dtype, slicestops.dtype, fromstarts.dtype, fromstops.dtype]))(grid, block, (tooffsets, slicestarts, slicestops, sliceouterlen, fromstarts, fromstops, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_getitem_jagged_descend_b", tooffsets.dtype, slicestarts.dtype, slicestops.dtype, fromstarts.dtype, fromstops.dtype]))(grid, block, (tooffsets, slicestarts, slicestops, sliceouterlen, fromstarts, fromstops, scan_in_array, invocation_index, err_code))
// out["awkward_ListArray_getitem_jagged_descend_a", {dtype_specializations}] = None
// out["awkward_ListArray_getitem_jagged_descend_b", {dtype_specializations}] = None
// END PYTHON

enum class LISTARRAY_GETITEM_JAGGED_DESCEND_ERRORS {
  INN_LEN_ERR,  // message: "jagged slice inner length differs from array inner length"
};

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_ListArray_getitem_jagged_descend_a(
    T* tooffsets,
    const C* slicestarts,
    const U* slicestops,
    int64_t sliceouterlen,
    const V* fromstarts,
    const W* fromstops,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < sliceouterlen) {
      int64_t slicecount = (int64_t)(slicestops[thread_id] -
                                    slicestarts[thread_id]);
      int64_t count = (int64_t)(fromstops[thread_id] -
                                fromstarts[thread_id]);
      if (slicecount != count) {
        RAISE_ERROR(LISTARRAY_GETITEM_JAGGED_DESCEND_ERRORS::INN_LEN_ERR)
      }
      scan_in_array[thread_id] = (T)count;
    }
  }
}

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_ListArray_getitem_jagged_descend_b(
    T* tooffsets,
    const C* slicestarts,
    const U* slicestops,
    int64_t sliceouterlen,
    const V* fromstarts,
    const W* fromstops,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (sliceouterlen == 0) {
      tooffsets[0] = 0;
    }
    else {
      tooffsets[0] = slicestarts[0];
    }

    if (thread_id < sliceouterlen) {
      int64_t slicecount = (int64_t)(slicestops[thread_id] -
                                    slicestarts[thread_id]);
      int64_t count = (int64_t)(fromstops[thread_id] -
                                fromstarts[thread_id]);
      if (slicecount != count) {
        RAISE_ERROR(LISTARRAY_GETITEM_JAGGED_DESCEND_ERRORS::INN_LEN_ERR)
      }
      tooffsets[thread_id + 1] = tooffsets[0] + scan_in_array[thread_id];
    }
  }
}
