// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (numvalid, slicestarts, slicestops, length, missing, missinglength, invocation_index, err_code) = args
//     scan_in_array = cupy.zeros(missinglength, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_getitem_jagged_numvalid_a", numvalid.dtype, slicestarts.dtype, slicestops.dtype, missing.dtype]))(grid, block, (numvalid, slicestarts, slicestops, length, missing, missinglength, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_getitem_jagged_numvalid_b", numvalid.dtype, slicestarts.dtype, slicestops.dtype, missing.dtype]))(grid, block, (numvalid, slicestarts, slicestops, length, missing, missinglength, scan_in_array, invocation_index, err_code))
// out["awkward_ListArray_getitem_jagged_numvalid_a", {dtype_specializations}] = None
// out["awkward_ListArray_getitem_jagged_numvalid_b", {dtype_specializations}] = None
// END PYTHON

enum class LISTARRAY_GETITEM_JAGGED_NUMVALID_ERRORS {
  STOP_LT_START,  // message: "jagged slice's stops[i] < starts[i]"
  OFF_GET_CON,    // message: "jagged slice's offsets extend beyond its content"
};

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_ListArray_getitem_jagged_numvalid_a(
    T* numvalid,
    const C* slicestarts,
    const U* slicestops,
    int64_t length,
    const V* missing,
    int64_t missinglength,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < length) {
      C slicestart = slicestarts[thread_id];
      U slicestop = slicestops[thread_id];

      if (slicestart != slicestop) {
        if (slicestop < slicestart) {
          RAISE_ERROR(LISTARRAY_GETITEM_JAGGED_NUMVALID_ERRORS::STOP_LT_START)
        }
        if (slicestop > missinglength) {
          RAISE_ERROR(LISTARRAY_GETITEM_JAGGED_NUMVALID_ERRORS::OFF_GET_CON)
        }
        for (int64_t j = slicestart;  j < slicestop;  j++) {
          scan_in_array[j] = missing[j] >= 0 ? 1 : 0;
        }
      }
    }
  }
}

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_ListArray_getitem_jagged_numvalid_b(
    T* numvalid,
    const C* slicestarts,
    const U* slicestops,
    int64_t length,
    const V* missing,
    int64_t missinglength,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    *numvalid = length > 0 ? scan_in_array[missinglength - 1] : 0;
  }
}
