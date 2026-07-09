// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (nextcarry, nextoffsets, outindex, index, offsets, outlength, invocation_index, err_code) = args
//     # Total element count from offsets[outlength] in the offsets pipeline.
//     length = int(offsets[int(outlength)].item())
//     scan_in_array = cupy.zeros(length, dtype=cupy.int64)
//     # Phase a/b need `length` threads; phase c needs `outlength + 1`.
//     block_size = block[0] if isinstance(block, tuple) else block
//     if block_size <= 0:
//         block_size = 1
//     grid_ab = ((length + block_size - 1) //block_size,) if length > 0 else (1,)
//     grid_c = ((int(outlength) + 1 + block_size - 1) //block_size,)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_reduce_next_64_a", nextcarry.dtype, nextoffsets.dtype, outindex.dtype, index.dtype, offsets.dtype]))(grid_ab, (block_size,), (nextcarry, nextoffsets, outindex, index, offsets, outlength, scan_in_array, invocation_index, err_code))
//     scan_in_array = cupy.cumsum(scan_in_array)
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_reduce_next_64_b", nextcarry.dtype, nextoffsets.dtype, outindex.dtype, index.dtype, offsets.dtype]))(grid_ab, (block_size,), (nextcarry, nextoffsets, outindex, index, offsets, outlength, scan_in_array, invocation_index, err_code))
//     cuda_kernel_templates.get_function(fetch_specialization(["awkward_IndexedArray_reduce_next_64_c", nextcarry.dtype, nextoffsets.dtype, outindex.dtype, index.dtype, offsets.dtype]))(grid_c, (block_size,), (nextcarry, nextoffsets, outindex, index, offsets, outlength, scan_in_array, invocation_index, err_code))
// out["awkward_IndexedArray_reduce_next_64_a", {dtype_specializations}] = None
// out["awkward_IndexedArray_reduce_next_64_b", {dtype_specializations}] = None
// out["awkward_IndexedArray_reduce_next_64_c", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_IndexedArray_reduce_next_64_a(
    T* nextcarry,
    C* nextoffsets,
    U* outindex,
    const V* index,
    const W* offsets,
    int64_t outlength,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t length = (int64_t)offsets[outlength];

    if (thread_id < length) {
      if (index[thread_id] >= 0) {
        scan_in_array[thread_id] = 1;
      }
    }
  }
}

template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_IndexedArray_reduce_next_64_b(
    T* nextcarry,
    C* nextoffsets,
    U* outindex,
    const V* index,
    const W* offsets,
    int64_t outlength,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t length = (int64_t)offsets[outlength];

    if (thread_id < length) {
      if (index[thread_id] >= 0) {
        nextcarry[scan_in_array[thread_id] - 1] = (T)index[thread_id];
        outindex[thread_id] = scan_in_array[thread_id] - 1;
      } else {
        outindex[thread_id] = -1;
      }
    }
  }
}

// Phase c: write nextoffsets per-bin boundary.
//   nextoffsets[b] = #valid (index >= 0) elements in [0, offsets[b]).
template <typename T, typename C, typename U, typename V, typename W>
__global__ void
awkward_IndexedArray_reduce_next_64_c(
    T* nextcarry,
    C* nextoffsets,
    U* outindex,
    const V* index,
    const W* offsets,
    int64_t outlength,
    int64_t* scan_in_array,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < outlength + 1) {
      int64_t boundary = (int64_t)offsets[thread_id];
      if (boundary == 0) {
        nextoffsets[thread_id] = 0;
      } else {
        nextoffsets[thread_id] = scan_in_array[boundary - 1];
      }
    }
  }
}
