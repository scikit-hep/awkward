// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

enum class INDEXEDARRAY_FLATTEN_NONE2EMPTY_ERRORS {
  OFF_OUT_OF_RANGE,  // message: "flattening offset out of range"
};

// BEGIN PYTHON
// def f(grid, block, args):
//     (outoffsets, outindex, outindexlength, offsets, offsetslength, invocation_index, err_code) = args
//     scan_in_array_k = cupy.zeros(outindexlength, dtype=cupy.int64)
//     scan_in_array_outoffsets = cupy.zeros(outindexlength + 1, dtype=cupy.int64)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_flatten_none2empty_a", outoffsets.dtype, outindex.dtype, offsets.dtype]))(grid, block, (outoffsets, outindex, outindexlength, offsets, offsetslength, scan_in_array_k, scan_in_array_outoffsets, invocation_index, err_code))
//     scan_in_array_k = cupy.cumsum(scan_in_array_k)
//     scan_in_array_outoffsets = cupy.cumsum(scan_in_array_outoffsets)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_flatten_none2empty_b", outoffsets.dtype, outindex.dtype, offsets.dtype]))(grid, block, (outoffsets, outindex, outindexlength, offsets, offsetslength, scan_in_array_k, scan_in_array_outoffsets, invocation_index, err_code))
// out["awkward_IndexedArray_flatten_none2empty_a", {dtype_specializations}] = None
// out["awkward_IndexedArray_flatten_none2empty_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_IndexedArray_flatten_none2empty_a(
    T* outoffsets,
    const C* outindex,
    int64_t outindexlength,
    const U* offsets,
    int64_t offsetslength,
    int64_t* scan_in_array_k,
    int64_t* scan_in_array_outoffsets,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    scan_in_array_outoffsets[0] = offsets[0];

    if (thread_id < outindexlength) {
      C idx = outindex[thread_id];
      if (idx < 0) {
        scan_in_array_k[thread_id] = 1;
      } else if (idx + 1 >= offsetslength) {
        RAISE_ERROR(INDEXEDARRAY_FLATTEN_NONE2EMPTY_ERRORS::OFF_OUT_OF_RANGE)
      } else {
          T count =
            offsets[idx + 1] - offsets[idx];
        scan_in_array_k[thread_id] = 1;
        scan_in_array_outoffsets[thread_id + 1] = count;
      }
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_IndexedArray_flatten_none2empty_b(
    T* outoffsets,
    const C* outindex,
    int64_t outindexlength,
    const U* offsets,
    int64_t offsetslength,
    int64_t* scan_in_array_k,
    int64_t* scan_in_array_outoffsets,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    outoffsets[0] = scan_in_array_outoffsets[0];

    if (thread_id < outindexlength) {
      C idx = outindex[thread_id];
      if (idx < 0) {
        outoffsets[scan_in_array_k[thread_id]] = scan_in_array_outoffsets[thread_id + 1];
      } else if (idx + 1 >= offsetslength) {
        RAISE_ERROR(INDEXEDARRAY_FLATTEN_NONE2EMPTY_ERRORS::OFF_OUT_OF_RANGE)
      } else {
        outoffsets[scan_in_array_k[thread_id]] = scan_in_array_outoffsets[thread_id + 1];
      }
    }
  }
}
