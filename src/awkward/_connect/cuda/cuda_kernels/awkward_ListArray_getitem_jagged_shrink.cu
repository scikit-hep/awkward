// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (tocarry, tosmalloffsets, tolargeoffsets, slicestarts, slicestops, length, missing, invocation_index, err_code) = args
//     if length > 0 and length < int(slicestops[length - 1]):
//         len_array = int(slicestops[length - 1])
//     else:
//         len_array = length
//     scan_in_array_k = cupy.zeros(len_array, dtype=cupy.int64)
//     scan_in_array_tosmalloffsets = cupy.zeros(length + 1, dtype=cupy.int64)
//     scan_in_array_tolargeoffsets = cupy.zeros(length + 1, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_getitem_jagged_shrink_a", tocarry.dtype, tosmalloffsets.dtype, tolargeoffsets.dtype, slicestarts.dtype, slicestops.dtype, missing.dtype]))(grid, block, (tocarry, tosmalloffsets, tolargeoffsets, slicestarts, slicestops, length, missing, scan_in_array_k, scan_in_array_tosmalloffsets, scan_in_array_tolargeoffsets, invocation_index, err_code))
//     scan_in_array_k = cupy.cumsum(scan_in_array_k)
//     scan_in_array_tosmalloffsets = cupy.cumsum(scan_in_array_tosmalloffsets)
//     scan_in_array_tolargeoffsets = cupy.cumsum(scan_in_array_tolargeoffsets)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_ListArray_getitem_jagged_shrink_b", tocarry.dtype, tosmalloffsets.dtype, tolargeoffsets.dtype, slicestarts.dtype, slicestops.dtype, missing.dtype]))(grid, block, (tocarry, tosmalloffsets, tolargeoffsets, slicestarts, slicestops, length, missing, scan_in_array_k, scan_in_array_tosmalloffsets, scan_in_array_tolargeoffsets, invocation_index, err_code))
// out["awkward_ListArray_getitem_jagged_shrink_a", {dtype_specializations}] = None
// out["awkward_ListArray_getitem_jagged_shrink_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V, typename W, typename X>
__global__ void
awkward_ListArray_getitem_jagged_shrink_a(
    T* tocarry,
    C* tosmalloffsets,
    U* tolargeoffsets,
    const V* slicestarts,
    const W* slicestops,
    int64_t length,
    const X* missing,
    int64_t* scan_in_array_k,
    int64_t* scan_in_array_tosmalloffsets,
    int64_t* scan_in_array_tolargeoffsets,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < length) {
      if (thread_id == 0) {
        scan_in_array_tosmalloffsets[0] = slicestarts[0];
        scan_in_array_tolargeoffsets[0] = slicestarts[0];
      }
      V slicestart = slicestarts[thread_id];
      W slicestop = slicestops[thread_id];
      if (slicestart != slicestop) {
        C smallcount = 0;
        for (int64_t j = slicestart + threadIdx.y;  j < slicestop;  j += blockDim.y) {
          if (missing[j] >= 0) {
            smallcount++;
          }
        }
        scan_in_array_k[thread_id + 1] = smallcount;
        scan_in_array_tosmalloffsets[thread_id + 1] = smallcount;
      }
      scan_in_array_tolargeoffsets[thread_id + 1] = slicestop - slicestart;
    }
  }
}

template <typename T, typename C, typename U, typename V, typename W, typename X>
__global__ void
awkward_ListArray_getitem_jagged_shrink_b(
    T* tocarry,
    C* tosmalloffsets,
    U* tolargeoffsets,
    const V* slicestarts,
    const W* slicestops,
    int64_t length,
    const X* missing,
    int64_t* scan_in_array_k,
    int64_t* scan_in_array_tosmalloffsets,
    int64_t* scan_in_array_tolargeoffsets,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (length == 0) {
      tosmalloffsets[0] = 0;
      tolargeoffsets[0] = 0;
    }
    else {
      tosmalloffsets[0] = slicestarts[0];
      tolargeoffsets[0] = slicestarts[0];
    }
    if (thread_id < length) {
      V slicestart = slicestarts[thread_id];
      W slicestop = slicestops[thread_id];
      int64_t k = scan_in_array_k[thread_id] - scan_in_array_k[0];
      if (slicestart != slicestop) {
        for (int64_t j = slicestart + threadIdx.y;  j < slicestop;  j += blockDim.y) {
          if (missing[j] >= 0) {
            tocarry[k] = j;
            k++;
          }
        }
        tosmalloffsets[thread_id + 1] = scan_in_array_tosmalloffsets[thread_id + 1];
      }
      else {
        tosmalloffsets[thread_id + 1] = scan_in_array_tosmalloffsets[thread_id];
      }
      tolargeoffsets[thread_id + 1] = scan_in_array_tolargeoffsets[thread_id + 1];
    }
  }
}
