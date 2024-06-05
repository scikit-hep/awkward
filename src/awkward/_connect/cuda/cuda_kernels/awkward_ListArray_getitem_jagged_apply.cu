// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tooffsets, tocarry, slicestarts, slicestops, sliceouterlen, sliceindex, sliceinnerlen, fromstarts, fromstops, contentlen, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(sliceouterlen + 1, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_getitem_jagged_apply_a", tooffsets.dtype, tocarry.dtype, slicestarts.dtype, slicestops.dtype, sliceindex.dtype, fromstarts.dtype, fromstops.dtype]))(grid, block, (tooffsets, tocarry, slicestarts, slicestops, sliceouterlen, sliceindex, sliceinnerlen, fromstarts, fromstops, contentlen, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_getitem_jagged_apply_b", tooffsets.dtype, tocarry.dtype, slicestarts.dtype, slicestops.dtype, sliceindex.dtype, fromstarts.dtype, fromstops.dtype]))(grid, block, (tooffsets, tocarry, slicestarts, slicestops, sliceouterlen, sliceindex, sliceinnerlen, fromstarts, fromstops, contentlen, scan_in_array, invocation_index, err_code))
// out["awkward_ListArray_getitem_jagged_apply_a", {dtype_specializations}] = None
// out["awkward_ListArray_getitem_jagged_apply_b", {dtype_specializations}] = None
// END PYTHON

enum class LISTARRAY_GETITEM_JAGGED_APPLY_ERRORS {
  JAG_STOP_LT_START,  // message: "jagged slice's stops[i] < starts[i]"
  OFF_GET_CON,        // message: "jagged slice's offsets extend beyond its content"
  STOP_LT_START,      // message: "stops[i] < starts[i]"
  STOP_GET_LEN,       // message: "stops[i] > len(content)"
  IND_OUT_OF_RANGE,   // message: "index out of range"
};

template <typename T, typename C, typename U, typename V, typename W, typename X, typename Y>
__global__ void
awkward_ListArray_getitem_jagged_apply_a(
    T* tooffsets,
    C* tocarry,
    const U* slicestarts,
    const V* slicestops,
    int64_t sliceouterlen,
    const W* sliceindex,
    int64_t sliceinnerlen,
    const X* fromstarts,
    const Y* fromstops,
    int64_t contentlen,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    scan_in_array[0] = 0;

    if (thread_id < sliceouterlen) {
      U slicestart = slicestarts[thread_id];
      V slicestop = slicestops[thread_id];

      if (slicestart != slicestop) {
        if (slicestop < slicestart) {
          RAISE_ERROR(LISTARRAY_GETITEM_JAGGED_APPLY_ERRORS::JAG_STOP_LT_START)
        }
        if (slicestop > sliceinnerlen) {
          RAISE_ERROR(LISTARRAY_GETITEM_JAGGED_APPLY_ERRORS::OFF_GET_CON)
        }
        int64_t start = (int64_t)fromstarts[thread_id];
        int64_t stop = (int64_t)fromstops[thread_id];
        if (stop < start) {
          RAISE_ERROR(LISTARRAY_GETITEM_JAGGED_APPLY_ERRORS::STOP_LT_START)
        }
        if (start != stop  &&  stop > contentlen) {
          RAISE_ERROR(LISTARRAY_GETITEM_JAGGED_APPLY_ERRORS::STOP_GET_LEN)
        }
        scan_in_array[thread_id + 1] = slicestop - slicestart;
      }
    }
  }
}

template <typename T, typename C, typename U, typename V, typename W, typename X, typename Y>
__global__ void
awkward_ListArray_getitem_jagged_apply_b(
    T* tooffsets,
    C* tocarry,
    const U* slicestarts,
    const V* slicestops,
    int64_t sliceouterlen,
    const W* sliceindex,
    int64_t sliceinnerlen,
    const X* fromstarts,
    const Y* fromstops,
    int64_t contentlen,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < sliceouterlen) {
      U slicestart = slicestarts[thread_id];
      V slicestop = slicestops[thread_id];
      tooffsets[thread_id] = (T)(scan_in_array[thread_id]);
      if (slicestart != slicestop) {
        if (slicestop < slicestart) {
          RAISE_ERROR(LISTARRAY_GETITEM_JAGGED_APPLY_ERRORS::JAG_STOP_LT_START)
        }
        if (slicestop > sliceinnerlen) {
          RAISE_ERROR(LISTARRAY_GETITEM_JAGGED_APPLY_ERRORS::OFF_GET_CON)
        }
        int64_t start = (int64_t)fromstarts[thread_id];
        int64_t stop = (int64_t)fromstops[thread_id];
        if (stop < start) {
          RAISE_ERROR(LISTARRAY_GETITEM_JAGGED_APPLY_ERRORS::STOP_LT_START)
        }
        if (start != stop  &&  stop > contentlen) {
          RAISE_ERROR(LISTARRAY_GETITEM_JAGGED_APPLY_ERRORS::STOP_GET_LEN)
        }
        int64_t count = stop - start;
        for (int64_t j = slicestart + threadIdx.y;  j < slicestop;  j += blockDim.y) {
          int64_t index = (int64_t) sliceindex[j];
          if (index < -count || index > count) {
            RAISE_ERROR(LISTARRAY_GETITEM_JAGGED_APPLY_ERRORS::IND_OUT_OF_RANGE)
          }
          if (index < 0) {
            index += count;
          }
          tocarry[scan_in_array[thread_id] + j - slicestart] = start + index;
        }
      }
    }
    tooffsets[sliceouterlen] = scan_in_array[sliceouterlen];
  }
}
