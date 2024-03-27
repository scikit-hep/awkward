// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tocarry, fromoffsets, offsetslength, fromstarts, fromstops, lencontent, invocation_index, err_code) = args
//     if offsetslength > 0:
//         len_array = int(fromoffsets[offsetslength - 1])
//     else:
//         len_array = 0
//     scan_in_array = cupy.zeros(len_array, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_broadcast_tooffsets_a", tocarry.dtype, fromoffsets.dtype, fromstarts.dtype, fromstops.dtype]))(grid, block, (tocarry, fromoffsets, offsetslength, fromstarts, fromstops, lencontent, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_broadcast_tooffsets_b", tocarry.dtype, fromoffsets.dtype, fromstarts.dtype, fromstops.dtype]))(grid, block, (tocarry, fromoffsets, offsetslength, fromstarts, fromstops, lencontent, scan_in_array, invocation_index, err_code))
// out["awkward_ListArray_broadcast_tooffsets_a", {dtype_specializations}] = None
// out["awkward_ListArray_broadcast_tooffsets_b", {dtype_specializations}] = None
// END PYTHON

enum class LISTARRAY_BROADCAST_TOOFFSETS_ERRORS {
  STOP_GET_LEN,      // message: "stops[i] > len(content)"
  OFF_DEC,  // message: "broadcast's offsets must be monotonically increasing"
  NESTED_ERR,  // message: "cannot broadcast nested list"
};

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_ListArray_broadcast_tooffsets_a(
    T* tocarry,
    const C* fromoffsets,
    int64_t offsetslength,
    const U* fromstarts,
    const V* fromstops,
    int64_t lencontent,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < offsetslength - 1) {
      int64_t start = (int64_t)fromstarts[thread_id];
      int64_t stop = (int64_t)fromstops[thread_id];

      if (start != stop  &&  stop > lencontent) {
        RAISE_ERROR(LISTARRAY_BROADCAST_TOOFFSETS_ERRORS::STOP_GET_LEN)
      }
      int64_t count = (fromoffsets[thread_id + 1] - fromoffsets[thread_id]);
      if (count < 0) {
        RAISE_ERROR(LISTARRAY_BROADCAST_TOOFFSETS_ERRORS::OFF_DEC)
      }
      if (stop - start != count) {
        RAISE_ERROR(LISTARRAY_BROADCAST_TOOFFSETS_ERRORS::NESTED_ERR)
      }
      for (int64_t j = start;  j < stop;  j++) {
        scan_in_array[fromoffsets[thread_id] + j - start] = 1;
      }
    }
  }
}

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_ListArray_broadcast_tooffsets_b(
    T* tocarry,
    const C* fromoffsets,
    int64_t offsetslength,
    const U* fromstarts,
    const V* fromstops,
    int64_t lencontent,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < offsetslength - 1) {
      int64_t start = (int64_t)fromstarts[thread_id];
      int64_t stop = (int64_t)fromstops[thread_id];

      if (start != stop  &&  stop > lencontent) {
        RAISE_ERROR(LISTARRAY_BROADCAST_TOOFFSETS_ERRORS::STOP_GET_LEN)
      }
      int64_t count = (int64_t)(fromoffsets[thread_id + 1] - fromoffsets[thread_id]);
      if (count < 0) {
        RAISE_ERROR(LISTARRAY_BROADCAST_TOOFFSETS_ERRORS::OFF_DEC)
      }
      if (stop - start != count) {
        RAISE_ERROR(LISTARRAY_BROADCAST_TOOFFSETS_ERRORS::NESTED_ERR)
      }

      for (int64_t j = start;  j < stop;  j++) {
        tocarry[scan_in_array[fromoffsets[thread_id] + j - start] - 1] = (T)j;
      }
    }
  }
}
